"""Gemini transport for clinical evidence: global concurrency cap, plain and grounded calls."""

import os
import threading

from medical_potential.config import GEMINI_FLASH_PREVIEW_MODEL
from medical_potential.gemini_utils import gemini_api_call

from .utils import log

# All Gemini calls -- across every molecule and every inner worker thread --
# acquire this semaphore first, bounding simultaneous requests to avoid 429s.
GEMINI_MAX_CONCURRENT = int(os.getenv("GEMINI_MAX_CONCURRENT", "5"))
_gemini_semaphore = threading.Semaphore(GEMINI_MAX_CONCURRENT)


def call_gemini(prompt: str, max_tokens: int = 1024, temperature: float = 0.2,
                model: str = GEMINI_FLASH_PREVIEW_MODEL) -> str:
    """Plain (ungrounded) Gemini call gated by the global semaphore."""
    with _gemini_semaphore:
        return gemini_api_call(
            prompt=prompt, model=model, temperature=temperature, max_output_tokens=max_tokens,
        )


def call_gemini_grounded(prompt: str, max_tokens: int = 1024, temperature: float = 0.2,
                         model: str = GEMINI_FLASH_PREVIEW_MODEL) -> tuple[str, list[dict]]:
    """Gemini call with Google Search grounding. Returns (text, grounding_url_records)."""
    with _gemini_semaphore:
        text, grounding_urls = gemini_api_call(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_output_tokens=max_tokens,
            google_search=True,
            get_grounding_urls=True,
        )
    log.info(f"Grounded call returned {len(grounding_urls)} source URL(s)")
    return text, grounding_urls
