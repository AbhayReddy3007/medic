"""Unmet needs extraction from grounded web search for diabetes by country."""

from __future__ import annotations

import asyncio
import logging
import os
import traceback
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

from .prompts import (
    UNMET_NEEDS_WEB_STEP1_PROMPT,
    UNMET_NEEDS_WEB_STEP2_PROMPT,
    UNMET_NEEDS_WEB_STEP3_PROMPT,
)
from .utils import (
    _extract_token_usage,
    _normalize_unmet_need_output,
    _safe_json_parse,
)

load_dotenv()

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

GENAI_CLIENT = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


async def extract_unmet_needs_from_web_multi_turn(
    country: str, model_name: str, extract_token_usage: bool = False
) -> dict[str, Any]:
    """Extract T2D unmet needs using a 3-step progressive web search strategy.

    The strategy progresses from broad (epidemiological landscape) to focused
    (clinical/financial barriers) before synthesising grounded findings into
    the structured JSON schema — mirroring how a human analyst would build
    context before drawing conclusions.

    Args:
        country: Country name to scope unmet need extraction.
        model_name: Gemini model name to use.
        extract_token_usage: If True, include Gemini token usage in output.

    Returns:
        Dict containing flat unmet needs and grounding details
    """
    normalized_country = country.strip()
    if not normalized_country:
        return {
            "country": country,
            "disease": "diabetes",
            "success": False,
            "error": "country must be a non-empty string",
            "unmet_needs": [],
        }

    try:
        gemini_response = await _extract_web_multi_turn(normalized_country, model_name=model_name, extract_token_usage=extract_token_usage)

        raw_response = str(gemini_response.get("text", ""))
        token_usage = gemini_response.get("token_usage", {})
        grounding_sources = gemini_response.get("grounding_sources", [])

        parsed = _safe_json_parse(raw_response)
        flat_unmet_needs = _normalize_unmet_need_output(parsed)

        # Add the country to the unmet needs list
        for unmet_need in flat_unmet_needs:
            if isinstance(unmet_need, dict):
                unmet_need["country"] = country
                unmet_need["source"] = "web_search"

        return {
            "country": normalized_country,
            "disease": "diabetes",
            "source_type": "web_grounded_multi_turn",
            "success": True,
            "unmet_needs": flat_unmet_needs,
            "grounding_sources": grounding_sources,
            "raw_response": raw_response,
            "token_usage": {
                "input_tokens": token_usage.get("input_tokens"),
                "output_tokens": token_usage.get("output_tokens"),
                "total_tokens": token_usage.get("total_tokens"),
            }
            if extract_token_usage
            else None,
            "parse_error": len(flat_unmet_needs) == 0,
        }
    except Exception as exc:
        tb = traceback.format_exc()
        print("[UNMET_NEEDS_WEB][ERROR] extract_unmet_needs_from_web_multi_turn failed")
        print(tb)
        logger.exception("extract_unmet_needs_from_web_multi_turn failed for country=%s", normalized_country)
        return {
            "country": normalized_country,
            "disease": "diabetes",
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": tb,
            "unmet_needs": [],
        }


def _extract_grounding_sources(response: Any) -> list[dict[str, str]]:
    """Extract grounded web sources from Gemini response metadata when available."""
    if not hasattr(response, "to_dict"):
        return []

    try:
        response_dict = response.to_dict()
    except Exception:
        return []

    sources: list[dict[str, str]] = []
    seen: set[str] = set()

    candidates = response_dict.get("candidates", [])
    for candidate in candidates:
        grounding_metadata = candidate.get("groundingMetadata", {})
        grounding_chunks = grounding_metadata.get("groundingChunks", [])
        for chunk in grounding_chunks:
            web = chunk.get("web", {})
            uri = str(web.get("uri", "")).strip()
            title = str(web.get("title", "")).strip()
            if not uri or uri in seen:
                continue
            seen.add(uri)
            sources.append({"title": title, "url": uri})

    return sources


def _sum_token_usage(usages: list[dict[str, Any]]) -> dict[str, int]:
    """Sum token usage dictionaries across multiple model calls."""
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }

    for usage in usages:
        if not isinstance(usage, dict):
            continue
        for key in totals:
            value = usage.get(key)
            if isinstance(value, int):
                totals[key] += value
            elif isinstance(value, float):
                totals[key] += int(value)

    return totals


# ---------------------------------------------------------------------------
# 3-Step Multi-Turn Web Search Strategy
# ---------------------------------------------------------------------------

def _sync_multi_turn_grounded_gemini_call(model: str, country: str) -> dict[str, Any]:
    """Execute a 3-step multi-turn Gemini research loop with Google Search grounding.

    Step 1: Broad epidemiological & public-health landscape (grounded).
    Step 2: Focused clinical, therapeutic, and financial barrier deep-dive (grounded).
    Step 3: Synthesis into structured JSON - no web search, pure reasoning over history.
    """
    search_tool = types.Tool(google_search=types.GoogleSearch())
    conversation_history: list[types.Content] = []

    # -- Step 1: Landscape Discovery ------------------------------------------
    logger.info("[MULTI_TURN][Step 1/3] Epidemiological landscape for %s", country)
    step1_prompt = UNMET_NEEDS_WEB_STEP1_PROMPT.format(country=country)
    conversation_history.append(
        types.Content(role="user", parts=[types.Part.from_text(text=step1_prompt)])
    )
    response1 = GENAI_CLIENT.models.generate_content(
        model=model,
        contents=conversation_history,
        config=types.GenerateContentConfig(tools=[search_tool], temperature=0),
    )
    conversation_history.append(response1.candidates[0].content)
    logger.info("[MULTI_TURN] Step 1 complete.")

    # -- Step 2: Clinical / Financial Barrier Deep-Dive -----------------------
    logger.info("[MULTI_TURN][Step 2/3] Clinical and financial barriers for %s", country)
    step2_prompt = UNMET_NEEDS_WEB_STEP2_PROMPT.format(country=country)
    conversation_history.append(
        types.Content(role="user", parts=[types.Part.from_text(text=step2_prompt)])
    )
    response2 = GENAI_CLIENT.models.generate_content(
        model=model,
        contents=conversation_history,
        config=types.GenerateContentConfig(tools=[search_tool], temperature=0),
    )
    conversation_history.append(response2.candidates[0].content)
    logger.info("[MULTI_TURN] Step 2 complete.")

    # -- Step 3: JSON Synthesis (no search tool) ------------------------------
    logger.info("[MULTI_TURN][Step 3/3] Synthesising findings into JSON for %s", country)
    step3_prompt = UNMET_NEEDS_WEB_STEP3_PROMPT.format(country=country)
    conversation_history.append(
        types.Content(role="user", parts=[types.Part.from_text(text=step3_prompt)])
    )
    final_response = GENAI_CLIENT.models.generate_content(
        model=model,
        contents=conversation_history,
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
        ),
    )
    logger.info("[MULTI_TURN] Step 3 complete.")

    # Grounding sources are present on step 1 & 2; collect both.
    grounding_sources: list[dict[str, str]] = []
    for resp in (response1, response2):
        grounding_sources.extend(_extract_grounding_sources(resp))

    # Deduplicate by URL.
    seen_urls: set[str] = set()
    unique_sources: list[dict[str, str]] = []
    for src in grounding_sources:
        url = src.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_sources.append(src)

    total_token_usage = _sum_token_usage(
        [
            _extract_token_usage(response1),
            _extract_token_usage(response2),
            _extract_token_usage(final_response),
        ]
    )

    return {
        "text": (final_response.text or "").strip(),
        "token_usage": total_token_usage,
        "grounding_sources": unique_sources,
    }


async def _extract_web_multi_turn(country: str, model_name: str, extract_token_usage: bool = False) -> dict[str, Any]:
    """Run 3-step multi-turn grounded Gemini extraction."""
    try:
        return await asyncio.to_thread(_sync_multi_turn_grounded_gemini_call, model_name, country)
    except Exception as exc:
        tb = traceback.format_exc()
        print("[UNMET_NEEDS_WEB][ERROR] _extract_web_multi_turn failed")
        print(tb)
        logger.exception("_extract_web_multi_turn failed")
        raise RuntimeError(f"Gemini multi-turn extraction failed: {type(exc).__name__}: {exc}") from exc

