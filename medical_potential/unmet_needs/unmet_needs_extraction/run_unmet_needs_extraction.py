"""Run unmet needs extraction from SoC (US/UK/Germany) and web (multiple countries).

Usage:
    python -m market_potential.unmet_needs.unmet_needs_extraction.run_unmet_needs_extraction

Outputs a single CSV to market_potential/unmet_needs/outputs/unmet_needs_all_countries.csv
with a `country` column identifying the source of each row.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.cloud import bigquery
from medical_potential.config import (
    BQ_DATASET_ID,
    PROJECT_ID,
    UNMET_NEEDS_TABLE,
)
from medical_potential.gcp_utils import get_bq_client

# Add project root to sys.path to support direct script execution
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from .unmet_needs_extraction_from_soc import extract_unmet_needs_from_soc
from .unmet_needs_extraction_from_web import extract_unmet_needs_from_web_multi_turn

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

GENAI_CLIENT = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Keep data/output paths anchored to unmet_needs/ after relocating this script.
BASE_DIR = Path(__file__).resolve().parent.parent

# Model names for different extraction types
# Model names
# gemini-3.1-pro-preview
# gemini-2.5-flash
PDF_EXTRACTION_MODEL_NAME = "gemini-3-flash-preview"
WEB_EXTRACTION_MODEL_NAME = "gemini-3-flash-preview"

# Country -> SoC documents directory mapping.
# All PDFs found under each directory are sent together in one extraction call.
SOC_COUNTRY_PDF_DIRS: dict[str, str] = {
    "US": str(BASE_DIR / "soc_documents" / "US"),
    "UK": str(BASE_DIR / "soc_documents" / "UK"),
    "Germany": str(BASE_DIR / "soc_documents" / "Germany"),
    "France": str(BASE_DIR / "soc_documents" / "France"),
}

# Output directory
OUTPUTS_DIR = BASE_DIR / "outputs"

# Countries for web-grounded extraction
WEB_COUNTRIES = [
    "China",
    "India",
    "Brazil",
    "Australia",
    "Russia",
    "Japan",
    "Korea",
]

# BigQuery Configuration
TABLE_ID = f"{PROJECT_ID}.{BQ_DATASET_ID}.{UNMET_NEEDS_TABLE}"


# ---------------------------------------------------------------------------
# Main extraction coroutine
# ---------------------------------------------------------------------------

async def run_extraction() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    all_frames: list[pd.DataFrame] = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0

    # ------------------------------------------------------------------
    # 1. SoC extraction - US, UK, Germany
    # ------------------------------------------------------------------
    if not SOC_COUNTRY_PDF_DIRS:
        logger.warning("SOC_COUNTRY_PDF_DIRS is empty; skipping SoC extraction.")
    else:
        logger.info("Running SoC extraction for countries: %s", list(SOC_COUNTRY_PDF_DIRS.keys()))
    soc_country_pdf_paths: dict[str, list[str]] = {}
    for country, pdf_dir in SOC_COUNTRY_PDF_DIRS.items():
        dir_path = Path(pdf_dir)
        if not dir_path.exists() or not dir_path.is_dir():
            logger.error("SoC directory not found for %s: %s", country, dir_path)
            continue
        pdf_paths = sorted(str(p) for p in dir_path.glob("*.pdf"))
        if not pdf_paths:
            logger.warning("No PDF files found for %s in %s", country, dir_path)
            continue
        soc_country_pdf_paths[country] = pdf_paths

    if not soc_country_pdf_paths:
        logger.warning("No valid SoC PDF directories found; skipping SoC extraction.")

    if soc_country_pdf_paths:
        soc_tasks = [
            extract_unmet_needs_from_soc(
                soc_document_path=pdf_paths,
                model_name=PDF_EXTRACTION_MODEL_NAME,
                country=country,
                extract_token_usage=True,
            )
            for country, pdf_paths in soc_country_pdf_paths.items()
        ]
        soc_results = await asyncio.gather(*soc_tasks, return_exceptions=True)

        for country, result in zip(soc_country_pdf_paths.keys(), soc_results):
            if isinstance(result, Exception):
                logger.error("SoC extraction failed for %s: %s", country, result)
                continue
            if result.get("success"):
                unmet_needs = result.get("unmet_needs", [])
                logger.info("SoC extraction succeeded for %s: %d unmet needs found.", country, len(unmet_needs))
                # Country is added inside extract_unmet_needs_from_soc.
                df_soc = pd.DataFrame(unmet_needs)
                all_frames.append(df_soc)
                usage = result.get("token_usage") or {}
                total_input_tokens += usage.get("input_tokens") or 0
                total_output_tokens += usage.get("output_tokens") or 0
                total_tokens += usage.get("total_tokens") or 0
            else:
                logger.error("SoC extraction failed for %s: %s", country, result.get("error"))
    else:
        logger.warning("No SoC extraction tasks to run.")

    # ------------------------------------------------------------------
    # 2. Web extraction - one task per country, run concurrently
    # ------------------------------------------------------------------
    if not WEB_COUNTRIES:
        logger.warning("WEB_COUNTRIES is empty; skipping web extraction.")
    else:
        logger.info("Running web extraction for countries: %s", WEB_COUNTRIES)
        web_tasks = [
            extract_unmet_needs_from_web_multi_turn(
                country,
                model_name=WEB_EXTRACTION_MODEL_NAME,
                extract_token_usage=True,
            )
            for country in WEB_COUNTRIES
        ]
        web_results = await asyncio.gather(*web_tasks, return_exceptions=True)

        for country, result in zip(WEB_COUNTRIES, web_results):
            if isinstance(result, Exception):
                logger.error("Web extraction failed for %s: %s", country, result)
                continue
            if result.get("success"):
                unmet_needs = result.get("unmet_needs", [])
                logger.info("%s: %d unmet needs found.", country, len(unmet_needs))
                df_web = pd.DataFrame(unmet_needs)
                all_frames.append(df_web)
                usage = result.get("token_usage") or {}
                total_input_tokens += usage.get("input_tokens") or 0
                total_output_tokens += usage.get("output_tokens") or 0
                total_tokens += usage.get("total_tokens") or 0
            else:
                logger.error("Web extraction failed for %s: %s", country, result.get("error"))

    # ------------------------------------------------------------------
    # 3. Concatenate all results
    # ------------------------------------------------------------------
    if not all_frames:
        logger.warning("No results collected - output file not written.")
        return

    combined = pd.concat(all_frames, ignore_index=True)

    # ------------------------------------------------------------------
    # 4. Add created_at_utc timestamp
    # ------------------------------------------------------------------
    combined["created_at_utc"] = pd.Timestamp.now(tz="UTC")
    combined["inclusion_flag"] = "N"
    
    # ------------------------------------------------------------------
    # 5. Delete existing rows for countries in current run
    # ------------------------------------------------------------------
    countries_to_refresh = sorted(combined["country"].dropna().astype(str).unique().tolist())

    if countries_to_refresh:
        try:
            client = get_bq_client()
            delete_query = f"DELETE FROM `{TABLE_ID}` WHERE country IN UNNEST(@countries)"
            delete_job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ArrayQueryParameter("countries", "STRING", countries_to_refresh)
                ]
            )
            delete_job = client.query(delete_query, job_config=delete_job_config)
            delete_job.result()
            logger.info("Deleted existing rows for countries: %s", countries_to_refresh)
        except Exception as e:
            logger.error("Failed to delete existing country rows from BigQuery: %s", e)
            raise
    else:
        logger.warning("No countries found in combined dataframe; skipping delete step.")

    # ------------------------------------------------------------------
    # 6. Push data to BigQuery
    # ------------------------------------------------------------------
    try:
        client = get_bq_client()
        
        load_schema = [
            bigquery.SchemaField("unmet_need", "STRING"),
            bigquery.SchemaField("category", "STRING"),
            bigquery.SchemaField("evidence", "STRING"),
            bigquery.SchemaField("country", "STRING"),
            bigquery.SchemaField("source", "STRING"),
            bigquery.SchemaField("inclusion_flag", "STRING"),
            bigquery.SchemaField("created_at_utc", "TIMESTAMP"),
        ]
        
        job_config = bigquery.LoadJobConfig(
            schema=load_schema,
                    write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        )
        
        load_job = client.load_table_from_dataframe(
            combined,
            TABLE_ID,
            job_config=job_config,
        )
        load_job.result()
        
        logger.info("Successfully pushed %d rows to BigQuery table %s", load_job.output_rows, TABLE_ID)
    except Exception as e:
        logger.error("Failed to push data to BigQuery: %s", e)
        raise

    # ------------------------------------------------------------------
    # 7. Print total token usage
    # ------------------------------------------------------------------
    print("\n=== Total Token Usage ===")
    print(f"  Input tokens  : {total_input_tokens:,}")
    print(f"  Output tokens : {total_output_tokens:,}")
    print(f"  Total tokens  : {total_tokens:,}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(run_extraction())
