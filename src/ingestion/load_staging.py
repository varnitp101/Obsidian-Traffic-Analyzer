"""
Module: load_staging.py
Description: Idempotent staging loader that reads raw STATS19 CSVs from data/raw/stats19/
and bulk loads them into PostgreSQL staging tables (staging.collision, staging.vehicle, staging.casualty).
Logs operational metrics to ops.ingestion_log.
"""

import sys
import time
import logging
from pathlib import Path
from typing import Optional
import pandas as pd
from sqlalchemy import text

from src.db.connection import get_db_engine
from src.db.load import bulk_copy_df, log_ingestion_event, get_table_row_count

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ingestion.staging")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_STATS19_DIR = REPO_ROOT / "data" / "raw" / "stats19"


def get_table_columns(schema_name: str, table_name: str) -> list[str]:
    """Retrieve column names for a database table."""
    engine = get_db_engine()
    with engine.connect() as conn:
        cols = conn.execute(
            text(
                """
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_schema = :schema AND table_name = :table
                ORDER BY ordinal_position;
                """
            ),
            {"schema": schema_name, "table": table_name}
        ).fetchall()
    return [r[0].lower() for r in cols]


def load_csv_to_staging(
    csv_path: Path,
    table_name: str,
    year: int = 2022,
    truncate_first: bool = True
) -> int:
    """
    Load a raw STATS19 CSV file into staging schema with column filtering and ops logging.
    """
    run_id = f"stage_{table_name}_{year}_{int(time.time())}"
    if not csv_path.exists():
        err = f"Raw CSV file not found: {csv_path}"
        logger.error(err)
        log_ingestion_event(
            run_id=run_id,
            source_name=f"stats19_{table_name}_{year}",
            status="failed",
            records_extracted=0,
            file_path=str(csv_path),
            error_message=err
        )
        raise FileNotFoundError(err)

    file_size = csv_path.stat().st_size
    logger.info(f"Loading {csv_path.name} ({file_size / (1024*1024):.2f} MB) into staging.{table_name}...")

    # Read CSV as strings to prevent premature casting or truncation
    df = pd.read_csv(csv_path, dtype=str, low_memory=False)
    raw_rows = len(df)
    logger.info(f"Read {raw_rows:,} raw records from {csv_path.name}")

    # Standardize column headers
    df.columns = [c.strip().lower() for c in df.columns]

    # Map possible alternative column names across years (e.g. accident_index -> collision_index)
    col_mapping = {
        "accident_index": "collision_index",
        "accident_year": "collision_year",
        "accident_reference": "collision_ref_no",
        "accident_severity": "collision_severity",
    }
    df = df.rename(columns=col_mapping)

    # Filter to columns that exist in the staging table
    target_columns = get_table_columns("staging", table_name)
    available_cols = [c for c in df.columns if c in target_columns]
    df = df[available_cols]

    # Bulk load into PostgreSQL
    loaded_count = bulk_copy_df(
        df=df,
        schema_name="staging",
        table_name=table_name,
        truncate_first=truncate_first,
        run_id=run_id,
        source_name=f"stats19_{table_name}_{year}"
    )

    log_ingestion_event(
        run_id=run_id,
        source_name=f"stats19_{table_name}_{year}",
        status="success",
        records_extracted=loaded_count,
        file_path=str(csv_path),
        file_size_bytes=file_size
    )

    return loaded_count


def load_all_staging(year: int = 2022) -> dict:
    """
    Load collisions, vehicles, and casualties for the target year into staging.
    """
    year_dir = RAW_STATS19_DIR / str(year)
    results = {}

    mappings = [
        (f"dft-road-casualty-statistics-collision-{year}.csv", "collision"),
        (f"dft-road-casualty-statistics-vehicle-{year}.csv", "vehicle"),
        (f"dft-road-casualty-statistics-casualty-{year}.csv", "casualty"),
    ]

    for fname, table_name in mappings:
        csv_file = year_dir / fname
        count = load_csv_to_staging(csv_file, table_name, year=year, truncate_first=True)
        results[table_name] = count

    logger.info(f"All staging tables loaded for year {year}: {results}")
    return results


if __name__ == "__main__":
    results = load_all_staging(2022)
    print("\n--- Staging Load Summary (2022) ---")
    for tbl, count in results.items():
        print(f"  staging.{tbl}: {count:,} rows")
