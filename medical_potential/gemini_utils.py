"""Shared Gemini API transport helpers."""

from __future__ import annotations

import logging
import threading
import time

from google import genai
from google.genai import types

from medical_potential.config import GEMINI_SERVICE_TIER, PROJECT_ID

logger = logging.getLogger(__name__)

_stats_lock = threading.Lock()
gemini_call_stats: dict[tuple[str, str | None], dict[str, object]] = {}


def _new_model_stats() -> dict[str, object]:
    return {
        "service_tier": None,
        "input_token_count": 0,
        "output_token_count": 0,
        "thought_token_count": 0,
        "tool_token_count": 0,
        "total_token_count": 0,
        "total_num_of_calls": 0,
        "num_of_search_enabled_calls": 0,
        "num_of_search_queries": 0,
    }


def reset_gemini_call_stats() -> None:
    """Reset usage counters before a new dimension-molecule run."""
    with _stats_lock:
        gemini_call_stats.clear()


def get_gemini_call_stats() -> list[dict[str, object]]:
    """Return one usage snapshot per model for the current run."""
    with _stats_lock:
        return [
            {"model_used": model, **stats}
            for (model, _service_tier), stats in sorted(
                gemini_call_stats.items(),
                key=lambda item: (item[0][0], item[0][1] or ""),
            )
        ]


def _get_model_stats(model: str, service_tier: str | None) -> dict[str, object]:
    return gemini_call_stats.setdefault(
        (model, service_tier),
        {**_new_model_stats(), "service_tier": service_tier},
    )


def _record_gemini_response_stats(
    response,
    model: str,
    requested_service_tier: str | None,
    google_search: bool,
) -> None:
    sdk_http_response = getattr(response, "sdk_http_response", None)
    headers = getattr(sdk_http_response, "headers", None)
    actual_tier = headers.get("x-gemini-service-tier") if headers else None
    usage = getattr(response, "usage_metadata", None)
    traffic_type = getattr(usage, "traffic_type", None)
    traffic_type_value = getattr(traffic_type, "value", traffic_type)
    if traffic_type_value == "ON_DEMAND_FLEX":
        actual_tier = "flex"
    elif traffic_type_value == "ON_DEMAND_STANDARD":
        actual_tier = "standard"
    input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
    candidate_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
    thought_tokens = int(getattr(usage, "thoughts_token_count", 0) or 0)
    tool_tokens = int(getattr(usage, "tool_use_prompt_token_count", 0) or 0)
    total_tokens = int(getattr(usage, "total_token_count", 0) or 0)
    search_query_values: set[str] = set()
    for candidate in getattr(response, "candidates", None) or []:
        grounding_metadata = getattr(candidate, "grounding_metadata", None)
        for query in getattr(grounding_metadata, "web_search_queries", None) or []:
            if isinstance(query, str) and query.strip():
                search_query_values.add(query.strip())
    search_queries = len(search_query_values)

    service_tier = actual_tier or requested_service_tier
    with _stats_lock:
        stats = _get_model_stats(model, service_tier)
        stats["total_num_of_calls"] = int(stats["total_num_of_calls"]) + 1
        stats["num_of_search_enabled_calls"] = int(stats["num_of_search_enabled_calls"]) + int(
            google_search
        )
        stats["input_token_count"] = int(stats["input_token_count"]) + input_tokens
        stats["output_token_count"] = int(stats["output_token_count"]) + candidate_tokens
        stats["thought_token_count"] = int(stats["thought_token_count"]) + thought_tokens
        stats["tool_token_count"] = int(stats["tool_token_count"]) + tool_tokens
        stats["total_token_count"] = int(stats["total_token_count"]) + total_tokens
        stats["num_of_search_queries"] = int(stats["num_of_search_queries"]) + search_queries


def extract_grounding(response) -> dict[str, object]:
    """Return the response text, web chunks, and the text segments each chunk supports (by chunk position)."""
    chunks: list[dict[str, str | None] | None] = []
    supports: list[dict[str, object]] = []
    for candidate in getattr(response, "candidates", None) or []:
        metadata = getattr(candidate, "grounding_metadata", None)
        if metadata is None:
            continue
        for chunk in metadata.grounding_chunks or []:
            web = getattr(chunk, "web", None)
            chunks.append(
                {"title": web.title, "domain": web.domain, "url": web.uri} if web else None
            )
        for support in metadata.grounding_supports or []:
            segment = getattr(support, "segment", None)
            supports.append(
                {
                    "text": getattr(segment, "text", None) or "",
                    "chunk_indices": list(support.grounding_chunk_indices or []),
                }
            )
        break
    return {"text": response.text or "", "chunks": chunks, "supports": supports}


def extract_grounding_urls(response) -> list[dict[str, object]]:
    """Extract and resolve only grounding chunks referenced by supports."""
    grounding = extract_grounding(response)
    chunks = grounding["chunks"]
    supports = grounding["supports"]
    referenced: dict[int, list[str]] = {}
    for support in supports:
        text = str(support.get("text") or "")
        for index in support.get("chunk_indices", []):
            if isinstance(index, int):
                referenced.setdefault(index, []).append(text)

    records_by_url: dict[str, dict[str, object]] = {}
    for index, supported_text in referenced.items():
        if index < 0 or index >= len(chunks) or not chunks[index]:
            continue
        chunk = chunks[index]
        url = chunk.get("url")
        if not isinstance(url, str) or not url.strip():
            continue
        record = records_by_url.setdefault(
            url,
            {
                "url": url,
                "original_url": url,
                "title": chunk.get("title"),
                "domain": chunk.get("domain"),
                "status": "unresolved",
                "supported_text": [],
            },
        )
        record["supported_text"] = list(
            dict.fromkeys([*record["supported_text"], *supported_text])
        )

    if not records_by_url:
        return []

    from medical_potential.vertexai_resolve import resolve_urls_batch

    results = resolve_urls_batch(list(records_by_url))
    for resolved_url, original_url, status in results:
        record = records_by_url.get(original_url)
        if record is None:
            continue
        record["url"] = resolved_url or original_url
        record["original_url"] = original_url
        record["status"] = status
    return list(records_by_url.values())


MAX_RETRIES = 5
INITIAL_BACKOFF = 4.0
REQUEST_TIMEOUT_MS = 1_200_000


def gemini_api_call(
    prompt: str,
    model: str,
    temperature: float = 0,
    max_output_tokens: int | None = None,
    google_search: bool = False,
    response_mime_type: str | None = None,
    system_instruction: str | None = None,
    pdf_attachments: list[tuple[str, bytes]] | None = None,
    return_grounding: bool = False,
    get_grounding_urls: bool = False,
    thinking_level: str = "medium",
) -> str | tuple[str, dict[str, object]] | tuple[str, list[dict[str, object]]]:
    """Call Gemini with optional search, response formatting, and PDF inputs.

    With return_grounding=True, returns (text, extract_grounding(response)).
    With get_grounding_urls=True, returns (text, extract_grounding_urls(response)).
    """
    if return_grounding and get_grounding_urls:
        raise ValueError("return_grounding and get_grounding_urls cannot both be true")

    tools = (
        [
            types.Tool(
                google_search=types.GoogleSearch(),
            )
        ]
        if google_search
        else None
    )
    last_error = None
    contents: str | list[types.Content] = prompt
    if pdf_attachments:
        parts = [types.Part.from_text(text=prompt)]
        parts.extend(
            types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")
            for _, pdf_bytes in pdf_attachments
        )
        contents = [types.Content(role="user", parts=parts)]
    config_kwargs = {
        "temperature": temperature,
        "tools": tools,
        "thinking_config": types.ThinkingConfig(
            # Options: MINIMAL, LOW, MEDIUM, HIGH
            thinking_level=types.ThinkingLevel(thinking_level.upper()),
        ),
    }
    if max_output_tokens is not None:
        config_kwargs["max_output_tokens"] = max_output_tokens
    if response_mime_type:
        config_kwargs["response_mime_type"] = response_mime_type
    if system_instruction:
        config_kwargs["system_instruction"] = system_instruction
    config = types.GenerateContentConfig(**config_kwargs)

    last_error = None
    configured_service_tier = (GEMINI_SERVICE_TIER or "").strip().lower()
    request_modes = ("flex", "standard") if configured_service_tier == "flex" else ("standard",)
    for request_mode in request_modes:
        client = genai.Client(
            vertexai=True,
            project=PROJECT_ID,
            location="global",
            http_options=types.HttpOptions(
                api_version="v1",
                headers={
                    "X-Vertex-AI-LLM-Request-Type": "shared",
                    "X-Vertex-AI-LLM-Shared-Request-Type": request_mode,
                },
                timeout=REQUEST_TIMEOUT_MS,
            ),
        )
        retry_count = 0
        backoff_delay = INITIAL_BACKOFF
        while retry_count <= MAX_RETRIES:
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
                _record_gemini_response_stats(response, model, request_mode, google_search)
                usage = getattr(response, "usage_metadata", None)
                traffic_type = getattr(usage, "traffic_type", None)
                mode = getattr(traffic_type, "value", traffic_type) or request_mode
                print(f"\nGemini response served via mode: {mode}")
                if get_grounding_urls:
                    return response.text or "", extract_grounding_urls(response)
                if return_grounding:
                    return response.text or "", extract_grounding(response)
                return response.text or ""
            except Exception as error:
                last_error = error
                retry_count += 1
                if retry_count > MAX_RETRIES:
                    logger.warning(
                        "Gemini %s mode max retries exceeded: %s",
                        request_mode,
                        error,
                    )
                    if request_mode == "flex":
                        logger.warning("Falling back to Gemini standard mode")
                        break
                    raise

                logger.warning(
                    "Gemini %s request failed with %s; waiting %.1fs (retry %s/%s)...",
                    request_mode,
                    type(error).__name__,
                    backoff_delay,
                    retry_count,
                    MAX_RETRIES,
                )
                time.sleep(backoff_delay)
                backoff_delay *= 2

    if last_error is not None:
        raise last_error
    return ""
