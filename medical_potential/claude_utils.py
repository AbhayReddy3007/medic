"""Shared Claude-on-Vertex-AI transport helpers.

Mirrors gemini_utils.py: one `claude_api_call` entry point with retries and
per-model usage stats, using the Anthropic SDK's Vertex client (same GCP
project and credentials as the Gemini calls).

    from medical_potential.claude_utils import claude_api_call
    text = claude_api_call("Summarise ...", model=CLAUDE_SONNET_MODEL)
"""

from __future__ import annotations

import base64
import logging
import threading
import time
from typing import Any

import anthropic
from anthropic import AnthropicVertex

from medical_potential.config import CLAUDE_VERTEX_REGION, PROJECT_ID

logger = logging.getLogger(__name__)

MAX_RETRIES = 5
INITIAL_BACKOFF = 4.0
REQUEST_TIMEOUT_SECONDS = 1200.0
DEFAULT_MAX_TOKENS = 16_000
WEB_SEARCH_TOOL_TYPE = "web_search_20250305"

_RETRYABLE_ERRORS = (
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.RateLimitError,
    anthropic.InternalServerError,
)

_stats_lock = threading.Lock()
claude_call_stats: dict[str, dict[str, object]] = {}
_client: AnthropicVertex | None = None
_client_lock = threading.Lock()


def _new_model_stats() -> dict[str, object]:
    return {
        "input_token_count": 0,
        "output_token_count": 0,
        "cache_creation_input_token_count": 0,
        "cache_read_input_token_count": 0,
        "total_token_count": 0,
        "total_num_of_calls": 0,
        "num_of_search_enabled_calls": 0,
        "num_of_search_queries": 0,
    }


def reset_claude_call_stats() -> None:
    """Reset usage counters before a new dimension-molecule run."""
    with _stats_lock:
        claude_call_stats.clear()


def get_claude_call_stats() -> list[dict[str, object]]:
    """Return one usage snapshot per model for the current run."""
    with _stats_lock:
        return [
            {"model_used": model, **stats}
            for model, stats in sorted(claude_call_stats.items())
        ]


def _record_claude_response_stats(message, model: str, web_search: bool) -> None:
    usage = getattr(message, "usage", None)
    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0
    cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    server_tool_use = getattr(usage, "server_tool_use", None)
    search_requests = getattr(server_tool_use, "web_search_requests", 0) or 0

    with _stats_lock:
        stats = claude_call_stats.setdefault(model, _new_model_stats())
        stats["input_token_count"] += input_tokens
        stats["output_token_count"] += output_tokens
        stats["cache_creation_input_token_count"] += cache_creation
        stats["cache_read_input_token_count"] += cache_read
        stats["total_token_count"] += input_tokens + output_tokens + cache_creation + cache_read
        stats["total_num_of_calls"] += 1
        if web_search:
            stats["num_of_search_enabled_calls"] += 1
        stats["num_of_search_queries"] += search_requests


def _get_client() -> AnthropicVertex:
    """Lazily build one shared client (uses Application Default Credentials)."""
    global _client
    with _client_lock:
        if _client is None:
            _client = AnthropicVertex(
                project_id=PROJECT_ID,
                region=CLAUDE_VERTEX_REGION,
                timeout=REQUEST_TIMEOUT_SECONDS,
                max_retries=0,  # retries are handled below so they are logged
            )
        return _client


def extract_text(message) -> str:
    """Concatenate the text blocks of a response (skips thinking and tool blocks)."""
    return "".join(
        block.text for block in message.content if getattr(block, "type", None) == "text"
    )


def extract_citations(message) -> list[dict[str, Any]]:
    """Return unique web citations (url, title) from a web-search response."""
    seen: dict[str, dict[str, Any]] = {}
    for block in message.content:
        for citation in getattr(block, "citations", None) or []:
            url = getattr(citation, "url", None)
            if url and url not in seen:
                seen[url] = {"url": url, "title": getattr(citation, "title", None)}
        if getattr(block, "type", None) == "web_search_tool_result":
            results = getattr(block, "content", None)
            if isinstance(results, list):
                for result in results:
                    url = getattr(result, "url", None)
                    if url and url not in seen:
                        seen[url] = {"url": url, "title": getattr(result, "title", None)}
    return list(seen.values())


WEB_FETCH_TOOL = {
    "name": "web_fetch",
    "description": (
        "Fetch the text content of one web page or PDF by exact URL. Only URLs from the "
        "provided source list can be fetched. Returns the page text (truncated if long)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"url": {"type": "string", "description": "Exact URL from the source list."}},
        "required": ["url"],
    },
}
FETCH_TIMEOUT_SECONDS = 30
MAX_FETCH_CHARS = 30_000
_FETCH_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; moa-eval-fetcher/1.0)"}


def _norm_url(url: str | None) -> str:
    return (url or "").strip().rstrip("/").lower()


def fetch_page_text(url: str) -> tuple[str, str]:
    """Download a URL and return (final_url, text). Handles HTML and PDF."""
    import io

    import requests

    response = requests.get(url, headers=_FETCH_HEADERS, timeout=FETCH_TIMEOUT_SECONDS, allow_redirects=True)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "").lower()
    if "pdf" in content_type or response.url.lower().endswith(".pdf"):
        import pymupdf

        with pymupdf.open(stream=io.BytesIO(response.content), filetype="pdf") as doc:
            text = "\n".join(page.get_text() for page in doc)
    else:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
            tag.decompose()
        text = " ".join(soup.get_text(" ").split())
    return response.url, text[:MAX_FETCH_CHARS]


def _run_web_fetch(url: str, allowed: set[str], fetched: dict[str, dict[str, Any]]) -> str:
    if _norm_url(url) not in allowed:
        return "Error: URL is not in the provided source list; only listed URLs can be fetched."
    try:
        final_url, text = fetch_page_text(url)
    except Exception as error:  # network/parse failures are reported to the model, not raised
        return f"Error: could not fetch {url}: {type(error).__name__}: {error}"
    # Count both the requested URL and its post-redirect URL as retrieved.
    for retrieved in (url, final_url):
        fetched.setdefault(_norm_url(retrieved), {"url": retrieved, "title": None})
    return text or "Error: page contained no extractable text."


def claude_api_call(
    prompt: str,
    model: str,
    temperature: float = 0,
    max_output_tokens: int = DEFAULT_MAX_TOKENS,
    web_search: bool = False,
    max_web_searches: int = 5,
    allowed_domains: list[str] | None = None,
    system_instruction: str | None = None,
    pdf_attachments: list[tuple[str, bytes]] | None = None,
    thinking_budget_tokens: int | None = None,
    return_citations: bool = False,
    fetch_urls: list[str] | None = None,
    max_web_fetches: int = 10,
) -> str | tuple[str, list[dict[str, Any]]]:
    """Call Claude on Vertex AI with optional web search, PDF inputs and thinking.

    Args:
        prompt: User prompt.
        model: Vertex model id, e.g. CLAUDE_SONNET_MODEL.
        temperature: Ignored (API requires 1) when thinking is enabled.
        max_output_tokens: Must exceed thinking_budget_tokens when thinking is on.
        web_search: Enable the server-side web search tool (must be enabled for
            the model/region in Vertex).
        allowed_domains: Restrict web search to these domains (web_search only).
        system_instruction: Optional system prompt.
        pdf_attachments: List of (name, bytes) PDFs sent as document blocks.
        thinking_budget_tokens: Enables extended thinking with this budget (>=1024).
        return_citations: Return (text, extract_citations(message)).
        fetch_urls: Enables a client-side `web_fetch` tool restricted to exactly these
            URLs (no search). Citations returned are the pages actually fetched.
        max_web_fetches: Maximum fetch calls before the model must answer.
    """
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for _, pdf_bytes in pdf_attachments or []:
        content.append({
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": base64.standard_b64encode(pdf_bytes).decode("ascii"),
            },
        })

    request: dict[str, Any] = {
        "model": model,
        "max_tokens": max_output_tokens,
        "messages": [{"role": "user", "content": content}],
    }
    if system_instruction:
        request["system"] = system_instruction
    if web_search:
        tool: dict[str, Any] = {
            "type": WEB_SEARCH_TOOL_TYPE,
            "name": "web_search",
            "max_uses": max_web_searches,
        }
        if allowed_domains:
            tool["allowed_domains"] = list(allowed_domains)
        request["tools"] = [tool]
    if fetch_urls:
        request["tools"] = request.get("tools", []) + [WEB_FETCH_TOOL]
    if thinking_budget_tokens:
        if max_output_tokens <= thinking_budget_tokens:
            raise ValueError("max_output_tokens must be greater than thinking_budget_tokens")
        request["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget_tokens}
    else:
        request["temperature"] = temperature

    if fetch_urls:
        return _claude_fetch_loop(request, model, fetch_urls, max_web_fetches, return_citations)

    client = _get_client()
    retry_count = 0
    backoff_delay = INITIAL_BACKOFF
    while True:
        try:
            # Streaming avoids SDK timeouts on long, high-token requests.
            with client.messages.stream(**request) as stream:
                message = stream.get_final_message()
            _record_claude_response_stats(message, model, web_search)
            text = extract_text(message)
            if return_citations:
                return text, extract_citations(message)
            return text
        except _RETRYABLE_ERRORS as error:
            retry_count += 1
            if retry_count > MAX_RETRIES:
                logger.warning("Claude max retries exceeded: %s", error)
                raise
            logger.warning(
                "Claude request failed with %s; waiting %.1fs (retry %s/%s)...",
                type(error).__name__,
                backoff_delay,
                retry_count,
                MAX_RETRIES,
            )
            time.sleep(backoff_delay)
            backoff_delay *= 2


def _stream_with_retries(client: AnthropicVertex, request: dict[str, Any]):
    retry_count = 0
    backoff_delay = INITIAL_BACKOFF
    while True:
        try:
            with client.messages.stream(**request) as stream:
                return stream.get_final_message()
        except _RETRYABLE_ERRORS as error:
            retry_count += 1
            if retry_count > MAX_RETRIES:
                logger.warning("Claude max retries exceeded: %s", error)
                raise
            logger.warning("Claude request failed with %s; retry %s/%s", type(error).__name__, retry_count, MAX_RETRIES)
            time.sleep(backoff_delay)
            backoff_delay *= 2


def _claude_fetch_loop(
    request: dict[str, Any],
    model: str,
    fetch_urls: list[str],
    max_web_fetches: int,
    return_citations: bool,
):
    """Run the client-side web_fetch tool loop until the model returns a final answer."""
    client = _get_client()
    allowed = {_norm_url(url) for url in fetch_urls}
    fetched: dict[str, dict[str, Any]] = {}
    messages = list(request["messages"])
    fetch_count = 0
    while True:
        message = _stream_with_retries(client, {**request, "messages": messages})
        _record_claude_response_stats(message, model, web_search=True)
        tool_uses = [block for block in message.content if getattr(block, "type", None) == "tool_use"]
        if message.stop_reason != "tool_use" or not tool_uses:
            text = extract_text(message)
            return (text, list(fetched.values())) if return_citations else text
        messages.append({"role": "assistant", "content": message.content})
        results = []
        for block in tool_uses:
            if fetch_count >= max_web_fetches:
                output = "Error: fetch limit reached. Answer using the pages already retrieved."
            else:
                fetch_count += 1
                output = _run_web_fetch((block.input or {}).get("url", ""), allowed, fetched)
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
        messages.append({"role": "user", "content": results})
