#!/usr/bin/env python3
"""
Simple script to test Gemini API connectivity and rate limits.

Usage:
    python scripts/test_gemini.py
"""

import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx

from src.core.config import settings


async def test_gemini_api():
    """Test Gemini API with a simple request."""
    print("=" * 60)
    print("Gemini API Test")
    print("=" * 60)

    api_key = settings.google_api_key
    model = settings.gemini_model

    if not api_key:
        print("ERROR: GOOGLE_API_KEY not set in .env")
        return

    print(f"API Key: {api_key[:10]}...{api_key[-4:]}")
    print(f"Model: {model}")
    print()

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": "Say 'Hello' and nothing else."}]
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 10,
        },
    }

    print("Sending test request...")
    print()

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            url,
            headers={"Content-Type": "application/json"},
            json=payload,
        )

        print(f"Status Code: {response.status_code}")
        print()
        print("Response Headers:")
        for key, value in response.headers.items():
            if "rate" in key.lower() or "quota" in key.lower() or "retry" in key.lower():
                print(f"  {key}: {value}")
        print()

        if response.status_code == 200:
            data = response.json()
            content = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            print(f"SUCCESS! Response: {content}")

            # Show usage
            usage = data.get("usageMetadata", {})
            print(f"Tokens used: {usage.get('promptTokenCount', 0)} input, {usage.get('candidatesTokenCount', 0)} output")

        elif response.status_code == 429:
            print("RATE LIMITED!")
            print()
            print("Full response:")
            print(response.text[:2000])
            print()
            print("This usually means:")
            print("  1. You've exceeded your per-minute quota (15 RPM for free tier)")
            print("  2. You've exceeded your daily quota (1,500 requests/day for free)")
            print("  3. Your API key has restricted quotas")
            print()
            print("Solutions:")
            print("  - Wait a few minutes and try again")
            print("  - Check quotas at: https://console.cloud.google.com/apis/api/generativelanguage.googleapis.com/quotas")
            print("  - Upgrade to a paid tier for higher limits")

        elif response.status_code == 400:
            print("BAD REQUEST!")
            print(response.text[:1000])

        elif response.status_code == 403:
            print("FORBIDDEN - API key invalid or lacks permissions")
            print(response.text[:1000])

        elif response.status_code == 404:
            print(f"MODEL NOT FOUND: {model}")
            print("Try one of: gemini-2.0-flash, gemini-1.5-pro, gemini-1.5-flash")
            print(response.text[:500])

        else:
            print(f"UNEXPECTED ERROR: {response.status_code}")
            print(response.text[:1000])


if __name__ == "__main__":
    asyncio.run(test_gemini_api())