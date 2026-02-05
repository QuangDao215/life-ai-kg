"""
Services package - Business logic and external API clients.

This package contains:
- PubMed client for fetching biomedical literature
- Document chunking service for text segmentation
- LLM client abstraction for multiple providers
- Entity and relation extraction service
"""

from src.services.chunking import (
    ChunkData,
    ChunkingResult,
    ChunkingService,
    chunk_documents_batch,
    chunk_text,
    estimate_tokens,
)
from src.services.extraction import (
    BatchExtractionResult,
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
    ExtractionService,
    extract_from_text,
)
from src.services.llm_client import (
    BaseLLMClient,
    GeminiClient,
    LLMError,
    LLMMessage,
    LLMProvider,
    LLMResponse,
    MockLLMClient,
    get_llm_client,
)
from src.services.pubmed import (
    DEFAULT_QUERY,
    Author,
    PubMedAPIError,
    PubMedArticle,
    PubMedClient,
    PubMedError,
    PubMedParseError,
    PubMedRateLimitError,
    fetch_ambroxol_parkinson_articles,
)

__all__ = [
    # PubMed Client
    "PubMedClient",
    "Author",
    "PubMedArticle",
    "PubMedError",
    "PubMedAPIError",
    "PubMedParseError",
    "PubMedRateLimitError",
    "fetch_ambroxol_parkinson_articles",
    "DEFAULT_QUERY",
    # Chunking Service
    "ChunkingService",
    "ChunkData",
    "ChunkingResult",
    "chunk_text",
    "chunk_documents_batch",
    "estimate_tokens",
    # LLM Client
    "BaseLLMClient",
    "GeminiClient",
    "MockLLMClient",
    "LLMProvider",
    "LLMMessage",
    "LLMResponse",
    "LLMError",
    "get_llm_client",
    # Extraction Service
    "ExtractionService",
    "ExtractedEntity",
    "ExtractedRelation",
    "ExtractionResult",
    "BatchExtractionResult",
    "extract_from_text",
]
