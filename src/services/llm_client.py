"""
LLM client abstraction for entity and relation extraction.

This module provides a unified interface for interacting with Google Gemini
for knowledge graph extraction tasks.

Features:
- Async HTTP requests
- Retry logic with exponential backoff
- Response parsing and validation
- Mock client for testing

Requirements addressed:
- REQ-KG-01: LLM-based entity extraction
- REQ-KG-02: LLM-based relation extraction
"""

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    GEMINI = "gemini"
    MOCK = "mock"


# API endpoint
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class LLMMessage:
    """Represents a message in the conversation."""

    role: str  # "system", "user", or "assistant"
    content: str


@dataclass
class LLMResponse:
    """Response from an LLM API call."""

    content: str
    model: str
    provider: LLMProvider
    usage: dict[str, int] = field(default_factory=dict)
    raw_response: dict[str, Any] = field(default_factory=dict)

    @property
    def input_tokens(self) -> int:
        """Get input token count."""
        return self.usage.get("input_tokens", 0)

    @property
    def output_tokens(self) -> int:
        """Get output token count."""
        return self.usage.get("output_tokens", 0)

    @property
    def total_tokens(self) -> int:
        """Get total token count."""
        return self.input_tokens + self.output_tokens


# =============================================================================
# Exceptions
# =============================================================================


class LLMError(Exception):
    """Base exception for LLM client errors."""

    pass


class LLMRateLimitError(LLMError):
    """Raised when rate limited by API."""

    pass


class LLMAPIError(LLMError):
    """Raised when API returns an error response."""

    pass


class LLMParseError(LLMError):
    """Raised when response parsing fails."""

    pass


# =============================================================================
# Base LLM Client
# =============================================================================


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(self, model: str | None = None, timeout: float = 60.0):
        """
        Initialize LLM client.

        Args:
            model: Model identifier
            timeout: Request timeout in seconds
        """
        self.model = model
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "BaseLLMClient":
        """Async context manager entry."""
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        """Get the HTTP client."""
        if self._client is None:
            raise RuntimeError(
                "LLM client must be used as async context manager: "
                "async with Client() as client: ..."
            )
        return self._client

    @abstractmethod
    async def complete(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """
        Generate a completion from the LLM.

        Args:
            messages: Conversation messages
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens to generate

        Returns:
            LLMResponse with generated content
        """
        pass

    @abstractmethod
    def provider(self) -> LLMProvider:
        """Get the provider type."""
        pass


# =============================================================================
# Google Gemini Client
# =============================================================================


class GeminiClient(BaseLLMClient):
    """Client for Google Gemini API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
    ):
        """
        Initialize Gemini client.

        Args:
            api_key: Google API key (defaults to settings)
            model: Model name (defaults to settings.gemini_model)
            timeout: Request timeout
        """
        super().__init__(model=model or settings.gemini_model, timeout=timeout)
        self.api_key = api_key or settings.google_api_key

        if not self.api_key:
            raise ValueError("Google API key not configured. Set GOOGLE_API_KEY in .env")

    def provider(self) -> LLMProvider:
        """Get provider type."""
        return LLMProvider.GEMINI

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, LLMRateLimitError)),
        stop=stop_after_attempt(5),
        wait=wait_random_exponential(multiplier=1, min=2, max=30),
    )
    async def complete(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Generate completion using Google Gemini API."""
        # Build the API URL
        url = f"{GEMINI_API_URL}/{self.model}:generateContent?key={self.api_key}"

        headers = {
            "Content-Type": "application/json",
        }

        # Convert messages to Gemini format
        contents = []
        system_instruction_text = None

        for msg in messages:
            if msg.role == "system":
                system_instruction_text = msg.content
            else:
                # Gemini expects "model" for assistant responses
                role = "model" if msg.role == "assistant" else "user"
                contents.append({
                    "role": role,
                    "parts": [{"text": msg.content}]
                })

        # Build the payload
        payload = {
            "contents": contents,
            "generation_config": {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                # Force JSON mode for better KG extraction accuracy
                "response_mime_type": "application/json",
            },
        }

        # Add system instruction if present
        if system_instruction_text:
            payload["system_instruction"] = {
                "parts": [{"text": system_instruction_text}]
            }

        logger.debug("Gemini API request", model=self.model)

        response = await self.client.post(
            url,
            headers=headers,
            json=payload,
        )

        # Handle rate limits (429) specifically
        if response.status_code == 429:
            # Log full response for debugging
            error_body = response.text
            logger.error(
                "Gemini rate limit details",
                status_code=response.status_code,
                headers=dict(response.headers),
                body=error_body[:1000],
            )

            # Try to get retry-after header
            retry_after = response.headers.get("retry-after")
            if retry_after:
                wait_time = int(retry_after)
                logger.warning("Gemini rate limit hit", retry_after=wait_time)
            else:
                # Default to 10 seconds if no header (paid tier has higher limits)
                wait_time = 10
                logger.warning("Gemini rate limit hit, waiting 10s")

            # Wait before raising to let tenacity retry
            await asyncio.sleep(wait_time)
            raise LLMRateLimitError(f"Rate limit exceeded, waited {wait_time}s")

        # Also check for 503 (overloaded) which Gemini sometimes returns
        if response.status_code == 503:
            logger.warning("Gemini service overloaded (503), retrying...")
            await asyncio.sleep(30)
            raise LLMRateLimitError("Service overloaded (503)")

        if response.status_code != 200:
            error_text = response.text[:500]
            logger.error(
                "Gemini API error",
                status_code=response.status_code,
                response=error_text,
            )
            # Provide helpful error messages
            if response.status_code == 404:
                raise LLMAPIError(
                    f"Model '{self.model}' not found. "
                    "Valid models: gemini-2.5-flash, gemini-2.0-flash, gemini-1.5-pro"
                )
            elif response.status_code == 400:
                raise LLMAPIError(f"Bad request: {error_text}")
            elif response.status_code == 403:
                raise LLMAPIError("API key invalid or lacks permissions")
            else:
                raise LLMAPIError(f"API returned status {response.status_code}: {error_text}")

        data = response.json()

        # Extract text content from response
        # Gemini returns: {"candidates": [{"content": {"parts": [{"text": "..."}]}}]}
        content = ""
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            for part in parts:
                if "text" in part:
                    content += part["text"]

        # Extract usage metadata
        usage_metadata = data.get("usageMetadata", {})

        return LLMResponse(
            content=content,
            model=self.model,
            provider=LLMProvider.GEMINI,
            usage={
                "input_tokens": usage_metadata.get("promptTokenCount", 0),
                "output_tokens": usage_metadata.get("candidatesTokenCount", 0),
            },
            raw_response=data,
        )


# =============================================================================
# Mock Client (for testing)
# =============================================================================


class MockLLMClient(BaseLLMClient):
    """
    Mock LLM client for testing without API calls.

    Returns predefined responses based on the input text.
    """

    def __init__(self, model: str = "mock-model", timeout: float = 60.0):
        """Initialize mock client."""
        super().__init__(model=model, timeout=timeout)
        self._responses: list[str] = []
        self._call_count = 0

    def provider(self) -> LLMProvider:
        """Get provider type."""
        return LLMProvider.MOCK

    def set_responses(self, responses: list[str]) -> None:
        """Set predefined responses for testing."""
        self._responses = responses
        self._call_count = 0

    async def complete(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.0,  # noqa: ARG002 - Required by interface
        max_tokens: int = 4096,  # noqa: ARG002 - Required by interface
    ) -> LLMResponse:
        """Return mock response."""
        # If predefined responses exist, use them in order
        if self._responses:
            idx = min(self._call_count, len(self._responses) - 1)
            content = self._responses[idx]
            self._call_count += 1
        else:
            # Generate a default mock response based on context
            content = self._generate_default_response(messages)

        return LLMResponse(
            content=content,
            model=self.model,
            provider=LLMProvider.MOCK,
            usage={"input_tokens": 100, "output_tokens": 50},
            raw_response={},
        )

    def _generate_default_response(self, messages: list[LLMMessage]) -> str:
        """Generate a default mock extraction response."""
        # Check if this is an entity or relation extraction request
        user_content = ""
        for msg in messages:
            if msg.role == "user":
                user_content = msg.content
                break

        # Check for relation extraction first (prompt contains "Extract relations")
        if "extract relations" in user_content.lower():
            return json.dumps({
                "relations": [
                    {
                        "head": "GBA1",
                        "tail": "Parkinson's disease",
                        "label": "INCREASES_RISK",
                        "confidence": 0.95,
                        "evidence": "GBA1 mutations increase risk of Parkinson's disease",
                    },
                    {
                        "head": "glucocerebrosidase",
                        "tail": "α-synuclein",
                        "label": "INHIBITS",
                        "confidence": 0.85,
                        "evidence": "Reduced glucocerebrosidase activity leads to α-synuclein accumulation",
                    },
                ]
            })
        elif "entities" in user_content.lower():
            return json.dumps({
                "entities": [
                    {
                        "name": "Parkinson's disease",
                        "type": "Disease",
                        "normalized_name": "parkinsons_disease",
                    },
                    {
                        "name": "GBA1",
                        "type": "Gene",
                        "normalized_name": "gba1",
                    },
                    {
                        "name": "glucocerebrosidase",
                        "type": "Protein",
                        "normalized_name": "glucocerebrosidase",
                    },
                    {
                        "name": "α-synuclein",
                        "type": "Protein",
                        "normalized_name": "alpha_synuclein",
                    },
                ]
            })
        else:
            return json.dumps({"result": "mock response"})


# =============================================================================
# Factory Function
# =============================================================================


def get_llm_client(
    provider: str | LLMProvider | None = None,
    **kwargs,
) -> BaseLLMClient:
    """
    Factory function to get an LLM client.

    Args:
        provider: Provider name ("gemini", "mock")
        **kwargs: Additional arguments passed to client constructor

    Returns:
        Configured LLM client

    Example:
        client = get_llm_client("gemini")
        async with client:
            response = await client.complete(messages)
    """
    if provider is None:
        provider = settings.llm_extractor

    if isinstance(provider, str):
        provider = LLMProvider(provider.lower())

    if provider == LLMProvider.GEMINI:
        return GeminiClient(**kwargs)
    elif provider == LLMProvider.MOCK:
        return MockLLMClient(**kwargs)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
