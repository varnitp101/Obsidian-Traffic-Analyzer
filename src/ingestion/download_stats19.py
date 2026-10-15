"""
Module: download_stats19.py
Description: Robust, idempotent downloader for UK DfT STATS19 Road Safety Data.
Downloads collisions, vehicles, and casualties CSV files for specified year(s)
into data/raw/stats19/<year>/ without altering raw records.
"""

import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ingestion.stats19")

# Base URL for official UK DfT STATS19 CSV repository
DFT_BASE_URL = "https://data.dft.gov.uk/road-accidents-safety-data"

# Standard file name patterns on DfT repository
FILE_TYPES = ["collision", "vehicle", "casualty"]


def get_retry_session(
    retries: int = 5,
    backoff_factor: float = 1.5,
    status_forcelist: tuple = (429, 500, 502, 503, 504)
) -> requests.Session:
    """Create a requests session with exponential backoff and retry strategy."""
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def download_file(
    url: str,
    target_path: Path,
    session: requests.Session,
    chunk_size: int = 1024 * 1024  # 1MB chunks
) -> Dict[str, any]:
    """
    Downloads a remote file with streaming and error handling.
    Returns download metadata dictionary.
    """
    start_time = time.time()
    result = {
        "url": url,
        "target_path": str(target_path),
        "status": "failed",
        "bytes_downloaded": 0,
        "elapsed_seconds": 0.0,
        "error_message": None
    }

    try:
        logger.info(f"Initiating download: {url} -> {target_path}")
        response = session.get(url, stream=True, timeout=30)
        response.raise_for_status()

        target_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = target_path.with_suffix(".tmp")

        total_bytes = 0
        with open(temp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    total_bytes += len(chunk)

        # Atomic rename once download is completely flushed to disk
        if target_path.exists():
            target_path.unlink()
        temp_path.rename(target_path)

        elapsed = time.time() - start_time
        result.update({
            "status": "success",
            "bytes_downloaded": total_bytes,
            "elapsed_seconds": round(elapsed, 2)
        })
        logger.info(
            f"Successfully downloaded {target_path.name} "
            f"({total_bytes / (1024*1024):.2f} MB in {elapsed:.2f}s)"
        )
    except Exception as e:
        elapsed = time.time() - start_time
        result.update({
            "status": "failed",
            "elapsed_seconds": round(elapsed, 2),
            "error_message": str(e)
        })
        logger.error(f"Download failed for {url}: {e}")

    return result


def download_stats19_year(
    year: int = 2022,
    output_base_dir: Optional[Path] = None
) -> Dict[str, Dict[str, any]]:
    """
    Downloads all 3 STATS19 CSVs (collision, vehicle, casualty) for a specific year.
    Stores raw files in data/raw/stats19/<year>/
    """
    if output_base_dir is None:
        project_root = Path(__file__).resolve().parent.parent.parent
        output_base_dir = project_root / "data" / "raw" / "stats19" / str(year)
    else:
        output_base_dir = output_base_dir / str(year)

    output_base_dir.mkdir(parents=True, exist_ok=True)
    session = get_retry_session()
    run_summary = {}

    for file_type in FILE_TYPES:
        file_name = f"dft-road-casualty-statistics-{file_type}-{year}.csv"
        url = f"{DFT_BASE_URL}/{file_name}"
        destination = output_base_dir / file_name

        if destination.exists() and destination.stat().st_size > 0:
            logger.info(f"File already exists and is non-empty: {destination} (Skipping download)")
            run_summary[file_type] = {
                "url": url,
                "target_path": str(destination),
                "status": "cached",
                "bytes_downloaded": destination.stat().st_size,
                "elapsed_seconds": 0.0,
                "error_message": None
            }
            try:
                from src.db.load import log_ingestion_event
                log_ingestion_event(
                    run_id=f"raw_stats19_{file_type}_{year}_{int(time.time())}",
                    source_name=f"stats19_raw_{file_type}_{year}",
                    status="cached",
                    records_extracted=0,
                    file_path=str(destination),
                    file_size_bytes=destination.stat().st_size
                )
            except Exception as log_err:
                logger.debug(f"Could not log ingestion event to DB: {log_err}")
            continue

        result = download_file(url, destination, session)
        run_summary[file_type] = result
        try:
            from src.db.load import log_ingestion_event
            log_ingestion_event(
                run_id=f"raw_stats19_{file_type}_{year}_{int(time.time())}",
                source_name=f"stats19_raw_{file_type}_{year}",
                status=result["status"],
                records_extracted=0,
                file_path=str(destination),
                file_size_bytes=result["bytes_downloaded"],
                error_message=result.get("error_message")
            )
        except Exception as log_err:
            logger.debug(f"Could not log ingestion event to DB: {log_err}")

    return run_summary


if __name__ == "__main__":
    target_year = 2022
    if len(sys.argv) > 1:
        target_year = int(sys.argv[1])
    logger.info(f"Starting STATS19 ingestion for year: {target_year}")
    summary = download_stats19_year(target_year)
    logger.info("Ingestion complete. Summary:")
    for k, v in summary.items():
        logger.info(f"  [{k}] status={v['status']} bytes={v['bytes_downloaded']}")

