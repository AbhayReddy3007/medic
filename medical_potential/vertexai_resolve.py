"""
vertexai_resolve.py
--------------------
Fast, single-attempt resolver for Vertex AI grounding proxy URLs, tuned to
avoid Google-side rate limiting while still maximizing successful resolution.

Vertex AI grounding URLs have the form:
    https://vertexaisearch.cloud.google.com/grounding-api-redirect/...

They are opaque, temporary redirect tokens (they expire after a few days) -
Google does not expose the real destination anywhere in the metadata, so the
only way to get it is to follow the redirect over HTTP.

Resolution order
----------------
1. Extraction  - decode inline without any HTTP request (cheap, kept as a
                 fast path for non-vertexai proxies / other redirect styles;
                 it will essentially never fire for real vertexaisearch
                 tokens, since those aren't self-describing).
2. Direct pass-through - if the URL is already a real (non-proxy) URL,
                         return it immediately.
3. HTTP HEAD   - ONE attempt (+ one same-attempt retry only on a raw
                 connection error, never on timeout).
4. HTTP GET    - ONE fallback attempt (streamed, body never downloaded) if
                 HEAD didn't resolve it.
5. Failure     - return the original proxy URL with a descriptive status.

Design changes vs. the naive version
-------------------------------------
- No nested retry loops. The old version could retry GET twice, then HEAD
  twice, each with its own urllib3-level retries underneath - meaning a
  single dead/slow URL could block a thread for minutes. Now each URL has a
  hard, predictable worst-case wall-clock budget (~20s), and the typical
  case (a healthy redirect) resolves in well under a second.
- Persistent, thread-local sessions. The old code built a brand-new
  requests.Session (and TLS connection) for every single URL. Now each
  worker thread builds ONE session and reuses its connection pool /
  keep-alive across every URL it processes, which is both faster and less
  likely to look like abusive traffic to Google.
- A shared token-bucket rate limiter caps total outbound requests/second
  across all threads, so raising worker count doesn't turn into a request
  burst that gets you throttled.
- Worker count tuned down from 10 to 6. This is a starting point, not a law
  of physics - if you're seeing very few timeouts, you can raise
  DEFAULT_WORKERS or RATE_LIMIT_PER_SEC; if you still see connection errors,
  lower them further. There's no way to hit zero unresolved URLs from this
  script alone - a meaningful fraction of tokens will be genuinely expired
  by the time you resolve them, and no retry strategy fixes that.

Output tuple per URL:
    (resolved_url: str, original_vertexai_url: str, status: str)

Excel integration
-----------------
`resolve_excel(path, url_col, output_path)` reads the column `url_col`
from the first sheet, resolves every URL in parallel, then writes three
new columns back to the same (or a new) file:
    resolved_url | original_vertexai_url | resolution_status
"""

from __future__ import annotations

import base64
import logging
import re
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("vertexai_resolve")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VERTEXAI_PROXY_HOST = "vertexaisearch.cloud.google.com"

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

# Separate connect / read timeouts (seconds). Kept tight on purpose - a
# healthy redirect responds in well under a second; anything past this is
# almost always a dead token or a throttled connection, and waiting longer
# just burns thread time without improving the odds.
CONNECT_TIMEOUT = 4
READ_TIMEOUT = 6

# Worker threads for the batch resolver. Tuned down from 10 -> 6 as a
# starting point that avoids looking like a scraping burst to Google.
DEFAULT_WORKERS = 6

# Global cap on outbound requests/second across ALL worker threads combined,
# regardless of DEFAULT_WORKERS. This is what actually prevents rate
# limiting - raising thread count without this would just create bigger
# bursts.
RATE_LIMIT_PER_SEC = 8


# ---------------------------------------------------------------------------
# Token-bucket rate limiter (shared across all threads)
# ---------------------------------------------------------------------------
class _RateLimiter:
    """Simple thread-safe token bucket. Blocks the caller just enough to
    keep the aggregate request rate at or below `rate_per_sec`."""

    def __init__(self, rate_per_sec: float):
        self._rate = float(rate_per_sec)
        self._tokens = float(rate_per_sec)
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last
                self._last = now
                self._tokens = min(self._rate, self._tokens + elapsed * self._rate)
                if self._tokens >= 1:
                    self._tokens -= 1
                    return
                wait = (1 - self._tokens) / self._rate
            time.sleep(wait)


_RATE_LIMITER = _RateLimiter(RATE_LIMIT_PER_SEC)


# ---------------------------------------------------------------------------
# Thread-local session with retry adapter (built ONCE per worker thread,
# reused for every URL that thread handles - keep-alive instead of a fresh
# TLS handshake per URL)
# ---------------------------------------------------------------------------
def _build_session() -> requests.Session:
    session = requests.Session()
    # Only auto-retry on transient 5xx status codes. Connection errors and
    # timeouts are handled explicitly (once) in `_attempt`, so we don't stack
    # urllib3-level retries on top of our own retry logic.
    retry = Retry(
        total=1,
        connect=0,
        read=0,
        status=1,
        backoff_factor=0.3,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
    )
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=20, pool_connections=20)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(BROWSER_HEADERS)
    return session


_thread_local = threading.local()


def _get_thread_session() -> requests.Session:
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = _build_session()
        _thread_local.session = session
    return session


# ---------------------------------------------------------------------------
# Layer 1 - Extraction helpers (no HTTP request)
# ---------------------------------------------------------------------------

def _is_vertexai_proxy(url: str) -> bool:
    try:
        return VERTEXAI_PROXY_HOST in urllib.parse.urlparse(url).netloc
    except Exception:
        return False


def _extract_from_query_params(url: str) -> Optional[str]:
    try:
        parsed = urllib.parse.urlparse(url)
        params = urllib.parse.parse_qs(parsed.query)
        for key in ("url", "target", "redirect", "dest", "destination", "u"):
            for v in params.get(key, []):
                decoded = urllib.parse.unquote(v)
                if decoded.startswith(("http://", "https://")):
                    return decoded
    except Exception:
        pass
    return None


def _extract_from_path_base64(url: str) -> Optional[str]:
    try:
        parsed = urllib.parse.urlparse(url)
        segments = [s for s in parsed.path.split("/") if s]
        for segment in segments:
            for s in (segment, urllib.parse.unquote(segment)):
                padded = s + "=" * (-len(s) % 4)
                for decoder in (base64.b64decode, base64.urlsafe_b64decode):
                    try:
                        decoded_bytes = decoder(padded)
                        decoded_str = decoded_bytes.decode("utf-8", errors="replace")
                        match = re.search(r"https?://[^\s\"'<>]+", decoded_str)
                        if match:
                            candidate = match.group(0).rstrip(".,;)")
                            if VERTEXAI_PROXY_HOST not in candidate:
                                return candidate
                    except Exception:
                        continue
    except Exception:
        pass
    return None


def _extract_http_substring(url: str) -> Optional[str]:
    try:
        decoded = urllib.parse.unquote(url)
        for m in re.finditer(r"https?://[^\s\"'<>&]+", decoded):
            candidate = m.group(0).rstrip(".,;)")
            if VERTEXAI_PROXY_HOST not in candidate:
                return candidate
    except Exception:
        pass
    return None


def _attempt_extraction(url: str) -> Optional[str]:
    for fn in (_extract_from_query_params, _extract_from_path_base64, _extract_http_substring):
        result = fn(url)
        if result:
            logger.debug("Extracted via %s: %s", fn.__name__, result)
            return result
    return None


# ---------------------------------------------------------------------------
# Layer 3 - HTTP resolution (single attempt per method, hard time budget)
# ---------------------------------------------------------------------------

def _attempt(method: str, url: str, session: requests.Session):
    """
    One HTTP attempt, with exactly one immediate retry if (and only if) the
    failure was a raw connection error (flaky handshake) - never retried on
    timeout, since a timeout almost always means the token/proxy is dead or
    throttled and waiting again won't help.

    Returns (final_url, status_code, error_name). error_name is None on a
    successful HTTP round-trip (even if status_code is an error code).
    """
    kwargs = dict(allow_redirects=True, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
    fn = getattr(session, method)
    if method == "get":
        kwargs["stream"] = True  # never download the body, just get final url/status

    for attempt in (1, 2):
        _RATE_LIMITER.acquire()
        try:
            response = fn(url, **kwargs)
            try:
                return response.url, response.status_code, None
            finally:
                response.close()
        except requests.exceptions.ConnectionError:
            if attempt == 1:
                time.sleep(0.3)
                continue
            return None, None, "connection_error"
        except requests.exceptions.Timeout:
            return None, None, "timeout"
        except requests.exceptions.RequestException:
            return None, None, "request_error"

    return None, None, "connection_error"


def _resolve_via_http(url: str, session: requests.Session) -> tuple[Optional[str], str]:
    """Try HEAD once, then GET once. No further retries."""
    last_status = "failed"
    for method in ("head", "get"):
        final_url, status_code, err = _attempt(method, url, session)
        if err is not None:
            last_status = err
            continue
        if final_url and final_url != url:
            return final_url, f"resolved_{method}"
        if status_code == 200:
            return final_url or url, f"resolved_{method}"
        if status_code in (403, 404):
            return None, f"http_{status_code}"
        last_status = f"http_{status_code}"
    return None, last_status


# ---------------------------------------------------------------------------
# Core resolver (single URL)
# ---------------------------------------------------------------------------

def resolve_url(url: str) -> tuple[str, str, str]:
    """
    Resolve a single URL through the multi-layer strategy.

    Returns (resolved_url, original_vertexai_url, status), where status is
    one of: "direct", "extracted", "resolved_head", "resolved_get",
    "timeout", "http_403", "http_404", "connection_error", "request_error",
    "failed".
    """
    original = url

    if not url or not isinstance(url, str):
        return url, url, "failed"

    url = url.strip()

    if not _is_vertexai_proxy(url):
        return url, url, "direct"

    extracted = _attempt_extraction(url)
    if extracted:
        return extracted, original, "extracted"

    session = _get_thread_session()
    resolved, status = _resolve_via_http(url, session)

    if resolved:
        return resolved, original, status

    logger.warning("Resolution failed for %s [%s]", url, status)
    return original, original, status or "failed"


# ---------------------------------------------------------------------------
# Batch resolver
# ---------------------------------------------------------------------------

def resolve_urls_batch(
    urls: list[str],
    workers: int = DEFAULT_WORKERS,
) -> list[tuple[str, str, str]]:
    """
    Resolve a list of URLs in parallel using a ThreadPoolExecutor.
    Deduplicates internally: each unique URL is resolved only once.
    """
    clean_urls = [u if isinstance(u, str) and u.strip() else "" for u in urls]
    unique_urls = list(dict.fromkeys(u for u in clean_urls if u))

    logger.info(
        "Resolving %d unique URLs with %d workers (rate cap %.1f req/s)...",
        len(unique_urls), workers, RATE_LIMIT_PER_SEC,
    )

    cache: dict[str, tuple[str, str, str]] = {}
    start = time.monotonic()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_url = {executor.submit(resolve_url, u): u for u in unique_urls}
        for future in as_completed(future_to_url):
            original_url = future_to_url[future]
            try:
                result = future.result()
            except Exception as exc:
                logger.error("Unexpected error resolving %s: %s", original_url, exc)
                result = (original_url, original_url, "failed")
            cache[original_url] = result

    results: list[tuple[str, str, str]] = []
    for u in clean_urls:
        results.append(cache[u] if u else ("", "", "skipped"))

    total = len([u for u in clean_urls if u])
    resolved_count = sum(
        1 for _, _, s in results if s in ("direct", "extracted", "resolved_head", "resolved_get")
    )
    elapsed = time.monotonic() - start
    logger.info(
        "Resolved %d / %d URLs (%.0f%%) in %.1fs",
        resolved_count, total, 100 * resolved_count / max(total, 1), elapsed,
    )
    return results


# ---------------------------------------------------------------------------
# Excel integration
# ---------------------------------------------------------------------------

def resolve_excel(
    input_path: str,
    url_col: str = "url_vertexai",
    output_path: Optional[str] = None,
    sheet_name: int | str = 0,
    workers: int = DEFAULT_WORKERS,
) -> pd.DataFrame:
    if output_path is None:
        output_path = input_path

    logger.info("Reading Excel file: %s  (sheet=%s, column=%s)", input_path, sheet_name, url_col)
    df = pd.read_excel(input_path, sheet_name=sheet_name, engine="openpyxl")

    if url_col not in df.columns:
        raise ValueError(
            f"Column '{url_col}' not found in sheet. "
            f"Available columns: {list(df.columns)}"
        )

    urls = df[url_col].tolist()
    results = resolve_urls_batch(urls, workers=workers)

    resolved_urls, original_urls, statuses = zip(*results) if results else ([], [], [])

    df["resolved_url"] = list(resolved_urls)
    df["original_vertexai_url"] = list(original_urls)
    df["resolution_status"] = list(statuses)

    logger.info("Writing results to: %s", output_path)
    df.to_excel(output_path, index=False, engine="openpyxl")
    logger.info("Done. %d rows written.", len(df))

    status_counts = df["resolution_status"].value_counts().to_dict()
    logger.info("Resolution summary: %s", status_counts)

    return df


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Resolve Vertex AI proxy URLs in an Excel file."
    )
    parser.add_argument("input", help="Path to the input Excel file.")
    parser.add_argument("--url-col", default="url_vertexai")
    parser.add_argument("--output", default=None)
    parser.add_argument("--sheet", default=0)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=RATE_LIMIT_PER_SEC,
        help=f"Max requests/sec across all workers (default: {RATE_LIMIT_PER_SEC}).",
    )
    args = parser.parse_args()

    if args.rate_limit != RATE_LIMIT_PER_SEC:
        _RATE_LIMITER = _RateLimiter(args.rate_limit)

    sheet = args.sheet
    try:
        sheet = int(sheet)
    except ValueError:
        pass

    resolve_excel(
        input_path=args.input,
        url_col=args.url_col,
        output_path=args.output,
        sheet_name=sheet,
        workers=args.workers,
    )
