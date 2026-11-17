"""Label Expansion Opportunity — shared Gemini call utilities.

Single home for the two Gemini call wrappers used throughout this
pipeline:

- ``gemini_generate()``              — resilient Gemini call with optional
  Google Search grounding, retries, and ungrounded fallback.
- ``gemini_generate_with_timeout()`` — runs ``gemini_generate`` under a
  hard wall-clock timeout, with retry-on-timeout.

Every module in this package that talks to Gemini imports from here
instead of defining or re-exporting its own copy, so there is exactly
one implementation of each wrapper.
"""

from __future__ import annotations

import logging
import re
import threading

from medical_potential.config import GEMINI_FLASH_PREVIEW_MODEL
from medical_potential.gemini_utils import gemini_api_call

logger = logging.getLogger(__name__)


def gemini_generate(
    prompt: str,
    *,
    system_instruction: str = "",
    use_search: bool = True,
    allow_ungrounded_fallback: bool = True,
) -> str:
    """Calls Gemini with optional Google Search grounding.

    Retries transient errors with exponential backoff. An empty (but
    non-erroring) response is retried once more with the SAME config
    before moving on - e.g. an empty grounded response is retried with
    grounding still in place, not immediately treated as failed or
    switched to a different config. Only after that retry is also empty
    does this move to the next config (falling back to a plain,
    non-grounded call) - unless ``allow_ungrounded_fallback`` is False,
    in which case an empty/failed grounded response is left empty rather
    than retried without grounding.
    """
    configs: list[bool] = []
    if use_search:
        configs.append(True)
    if not use_search or allow_ungrounded_fallback:
        configs.append(False)

    last_err: Exception | None = None
    for i, use_search_config in enumerate(configs):
        try:
            text = gemini_api_call(
                prompt,
                model=GEMINI_FLASH_PREVIEW_MODEL,
                temperature=0,
                google_search=use_search_config,
                system_instruction=system_instruction or "Return ONLY valid JSON.",
            )
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
            if text:
                return text
        except Exception as exc:
            last_err = exc
            if i == 0 and len(configs) > 1:
                logger.info("[UTILS] Error with Search grounding (%s) — trying without grounding", exc)
                continue
            if i == 0 and not allow_ungrounded_fallback:
                logger.warning(
                    "[UTILS] Error with Search grounding (%s) and ungrounded fallback disabled - leaving empty",
                    exc,
                )
                return ""
            raise
        if i == 0 and len(configs) > 1:
            logger.info("[UTILS] Empty/failed response with Search grounding — retrying without grounding")
        elif i == 0 and not allow_ungrounded_fallback:
            logger.info("[UTILS] Empty response with Search grounding and ungrounded fallback disabled - leaving empty")
            return ""

    if last_err and allow_ungrounded_fallback:
        raise last_err
    return ""


def gemini_generate_with_timeout(
    prompt: str,
    *,
    system_instruction: str = "",
    use_search: bool = True,
    allow_ungrounded_fallback: bool = True,
    timeout_seconds: float,
    max_attempts: int = 2,
    log_context: str = "",
) -> str:
    """Runs ``gemini_generate`` in a background thread with a hard wall-clock
    timeout, retrying up to ``max_attempts`` times if it doesn't come back
    in time.

    Shared by any caller that batches Gemini calls so batches that
    legitimately need more time get it, without one hung batch blocking
    a whole run.

    ``allow_ungrounded_fallback`` is forwarded to ``gemini_generate`` as-is
    (see there): set it to ``False`` to leave an empty/failed grounded
    response empty instead of retrying it without Search grounding.

    Raises ``TimeoutError`` if every attempt times out, or re-raises
    whatever ``gemini_generate`` itself raised.
    """
    result_holder: dict = {}
    error_holder: dict = {}
    context_suffix = f" for {log_context}" if log_context else ""

    for attempt in range(1, max_attempts + 1):
        result_holder.clear()
        error_holder.clear()

        def _call_gemini():
            try:
                result_holder["text"] = gemini_generate(
                    prompt,
                    system_instruction=system_instruction,
                    use_search=use_search,
                    allow_ungrounded_fallback=allow_ungrounded_fallback,
                )
            except Exception as exc:  # noqa: BLE001
                error_holder["error"] = exc

        thread = threading.Thread(target=_call_gemini, daemon=True)
        thread.start()
        thread.join(timeout=timeout_seconds)

        if not thread.is_alive():
            break  # got a response (success or error) within the timeout

        if attempt < max_attempts:
            logger.warning(
                "[UTILS] Timeout (>%ss) on attempt %d/%d%s - retrying",
                timeout_seconds, attempt, max_attempts, context_suffix,
            )
        else:
            logger.warning(
                "[UTILS] Timeout (>%ss) on attempt %d/%d%s - giving up",
                timeout_seconds, attempt, max_attempts, context_suffix,
            )
            raise TimeoutError(f"Gemini call timed out after {max_attempts} attempt(s){context_suffix}")

    if "error" in error_holder:
        raise error_holder["error"]

    return result_holder.get("text", "")