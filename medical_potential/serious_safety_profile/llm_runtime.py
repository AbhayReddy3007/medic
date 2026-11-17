"""Two-Step Gemini Clinical Trial Extraction - Core Function

This module provides the core extraction logic used by the market potential agent.
"""
import os
import json
import asyncio
from dotenv import load_dotenv
from json_repair import repair_json
from medical_potential.gemini_utils import gemini_api_call

# Load environment variables
load_dotenv(override=True)


def safe_json_parse(json_str: str, context: str = "") -> dict | list | None:
    """Parse JSON with automatic repair using json_repair library.

    Args:
        json_str: The JSON string to parse
        context: Optional context for error messages

    Returns:
        Parsed JSON object/array, or None if parsing fails
    """
    if not json_str or not json_str.strip():
        return None

    # Clean up common issues
    json_str = json_str.strip()

    # Remove markdown code blocks if present
    if "```json" in json_str:
        json_str = json_str.split("```json")[1].split("```")[0].strip()
    elif "```" in json_str:
        parts = json_str.split("```")
        if len(parts) >= 2:
            json_str = parts[1].strip()

    # Find first { or [ if there's text before JSON
    json_start = -1
    for i, c in enumerate(json_str):
        if c in '{[':
            json_start = i
            break
    if json_start > 0:
        json_str = json_str[json_start:]

    # Try json_repair first (handles most issues automatically)
    try:
        repaired = repair_json(json_str, return_objects=True)
        if repaired is not None:
            return repaired
    except Exception as e:
        if context:
            print(f"  ⚠️ json_repair failed for {context}: {e}")

    # Fallback to standard json.loads
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    return None

# Configuration
BATCH_SIZE = 8
MAX_WORKERS = 8
RATE_LIMIT_DELAY = 5.0
MAX_RETRIES = 5
INITIAL_BACKOFF = 2.0

# Models
SEARCH_MODEL = "gemini-3-flash-preview"
EXTRACTION_MODEL = "gemini-3-flash-preview"


async def gemini_call_with_search(
    prompt: str,
    model: str,
    response_mime_type: str = None,
    temperature: float = 0,
) -> str:
    """Make a Gemini call with Google Search enabled.

    Args:
        prompt: The prompt to send to Gemini
        model: Model name (e.g., "gemini-2.5-flash")
        response_mime_type: Optional mime type (e.g., "application/json").
                           Note: Using response_mime_type with Google Search tools may cause errors.
    """
    return await asyncio.to_thread(
        gemini_api_call,
        prompt=prompt,
        model=model,
        temperature=temperature,
        google_search=True,
        response_mime_type=response_mime_type,
    )


async def gemini_call_without_search(
    prompt: str,
    model: str,
    response_mime_type: str = None,
    temperature: float = 0,
) -> str:
    """Make a Gemini call without Google Search enabled.

    Args:
        prompt: The prompt to send to Gemini
        model: Model name (e.g., "gemini-2.5-flash")
        response_mime_type: Optional mime type (e.g., "application/json").
    """
    return await asyncio.to_thread(
        gemini_api_call,
        prompt=prompt,
        model=model,
        temperature=temperature,
        google_search=False,
        response_mime_type=response_mime_type,
    )
