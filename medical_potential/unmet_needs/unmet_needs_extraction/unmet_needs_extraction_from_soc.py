"""Unmet needs extraction from standard-of-care (SoC) documents.

This module provides a callable function that:
1. Reads a local SoC document path provided at runtime.
2. Sends PDFs directly to Gemini.
3. Extracts patient-focused unmet need statements with supporting evidence.
"""

from __future__ import annotations

import asyncio
import logging
import os
import traceback
from pathlib import Path
from typing import Any, Union

from dotenv import load_dotenv
from google import genai
from google.genai import types

from .prompts import UNMET_NEEDS_EXTRACTION_PROMPT
from .utils import (
    _extract_token_usage,
    _normalize_unmet_need_output,
    _safe_json_parse,
)

load_dotenv()

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

GENAI_CLIENT = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def _build_unmet_need_prompt(document_name: str) -> str:
    return UNMET_NEEDS_EXTRACTION_PROMPT.format(document_name=document_name)


def _build_unmet_need_prompt_multi(document_names: list[str]) -> str:
    """Build prompt for multiple documents."""
    names_str = ", ".join(document_names)
    return UNMET_NEEDS_EXTRACTION_PROMPT.format(document_name=f"Multiple documents: {names_str}")


def _sync_gemini_pdf_call(prompt: str, pdf_bytes: Union[bytes, list[bytes]], model_name: str) -> dict[str, Any]:
    """Synchronous Gemini call wrapper using PDF document payload(s).
    
    Args:
        prompt: Prompt text.
        pdf_bytes: Single bytes object or list of bytes objects for multiple PDFs.
        model_name: Gemini model name to use.
    """
    # Normalize to list of bytes
    pdf_list = [pdf_bytes] if isinstance(pdf_bytes, bytes) else pdf_bytes
    
    # Build parts: text prompt + all PDFs
    parts = [types.Part.from_text(text=prompt)]
    parts.extend(
        types.Part.from_bytes(data=pdf, mime_type="application/pdf")
        for pdf in pdf_list
    )
    
    contents = [types.Content(role="user", parts=parts)]

    response = GENAI_CLIENT.models.generate_content(
        model=model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0
            ),
    )
    token_usage = _extract_token_usage(response)
    return {
        "text": (response.text or "").strip(),
        "token_usage": token_usage,
    }


async def _extract_pdf_with_gemini(prompt: str, pdf_bytes: Union[bytes, list[bytes]], model_name: str) -> dict[str, Any]:
    """Run Gemini extraction call with PDF attachment(s).
    
    Args:
        prompt: Prompt text.
        pdf_bytes: Single bytes object or list of bytes objects for multiple PDFs.
        model_name: Gemini model name to use.
    """
    try:
        return await asyncio.to_thread(_sync_gemini_pdf_call, prompt, pdf_bytes, model_name)
    except Exception as exc:
        tb = traceback.format_exc()
        print("[UNMET_NEEDS][ERROR] _extract_pdf_with_gemini failed")
        print(tb)
        logger.exception("_extract_pdf_with_gemini failed")
        raise RuntimeError(f"Gemini PDF extraction failed: {type(exc).__name__}: {exc}") from exc


async def extract_unmet_needs_from_soc(
    soc_document_path: Union[str, list[str]],
    model_name: str,
    country: str | None = None,
    extract_token_usage: bool = False,
) -> dict[str, Any]:
    """Extract patient-focused unmet needs from SoC document(s).

    Args:
        soc_document_path: Exact path to SoC PDF document, or list of paths for multiple documents.
        model_name: Gemini model name to use.
        country: Country label to attach to each extracted unmet need.
        extract_token_usage: If True, include Gemini token usage in output.

    Returns:
        dict with unmet need candidates and supporting evidence.
    """
    # Normalize input to list
    paths = [soc_document_path] if isinstance(soc_document_path, str) else soc_document_path
    path_objs = [Path(p) for p in paths]
    
    # Validate all paths
    for path in path_objs:
        if not path.exists():
            return {
                "document_paths": [str(p) for p in path_objs],
                "success": False,
                "error": f"File not found: {path}",
                "unmet_needs": [],
            }
        if path.suffix.lower() != ".pdf":
            return {
                "document_paths": [str(p) for p in path_objs],
                "success": False,
                "error": f"Unsupported file type for {path}. All inputs must be PDF documents.",
                "unmet_needs": [],
            }

    try:
        # Read all PDF bytes
        pdf_bytes_list = [p.read_bytes() for p in path_objs]
        document_names = [p.name for p in path_objs]
        
        # Build prompt
        if len(path_objs) == 1:
            prompt = _build_unmet_need_prompt(document_name=document_names[0])
        else:
            prompt = _build_unmet_need_prompt_multi(document_names)

        # Send to Gemini (single PDF or multiple PDFs)
        pdf_payload = pdf_bytes_list[0] if len(pdf_bytes_list) == 1 else pdf_bytes_list
        gemini_response = await _extract_pdf_with_gemini(prompt=prompt, pdf_bytes=pdf_payload, model_name=model_name)
        raw_response = str(gemini_response.get("text", ""))
        token_usage = gemini_response.get("token_usage", {})
        parsed = _safe_json_parse(raw_response)
        flat_unmet_needs = _normalize_unmet_need_output(parsed)
        for unmet_need in flat_unmet_needs:
            if isinstance(unmet_need, dict):
                unmet_need["country"] = country
                unmet_need["source"] = "soc_document"

        return {
            "document_paths": [str(p) for p in path_objs],
            "num_documents": len(path_objs),
            "document_type": "pdf",
            "success": True,
            "unmet_needs": flat_unmet_needs,
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
        print("[UNMET_NEEDS][ERROR] extract_unmet_needs_from_soc failed")
        print(tb)
        logger.exception("extract_unmet_needs_from_soc failed for %s", [str(p) for p in path_objs])
        return {
            "document_paths": [str(p) for p in path_objs],
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": tb,
            "unmet_needs": [],
        }
