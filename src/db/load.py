"""
Data Loading Utilities.
High-performance, idempotent bulk loading from pandas DataFrames / CSVs
into PostgreSQL schemas (staging, clean, mart) with telemetry logged to ops.ingestion_log.
"""

import os
import io
import time
import uuid
import logging
from typing import Optional, List
from pathlib import Path
import pandas as pd
from sqlalchemy import text

from src.db.connection import get_db_engine, get_db_connection

logger = logging.getLogger(__name__)


def log_ingestion_event(
    run_id: str,
    source_name: str,
    status: str,
    records_extracted: int,
    file_path: Optional[str] = None,
    file_size_bytes: Optional[int] = None,
    error_message: Optional[str] = None,
) -> None:
    """
    Log ingestion run status and metrics to ops.ingestion_log.
    """
    engine = get_db_engine()
    query = text(
        """
        INSERT INTO ops.ingestion_log (
            run_id, source_name, extraction_timestamp, status, 
            records_extracted, file_path, file_size_bytes, error_message
        )
        VALUES (
            :run_id, :source_name, CURRENT_TIMESTAMP, :status, 
            :records_extracted, :file_path, :file_size_bytes, :error_message
        )
        ON CONFLICT (run_id) DO UPDATE SET
            status = EXCLUDED.status,
            records_extracted = EXCLUDED.records_extracted,
            file_path = EXCLUDED.file_path,
            file_size_bytes = EXCLUDED.file_size_bytes,
            error_message = EXCLUDED.error_message,
            extraction_timestamp = CURRENT_TIMESTAMP;
        """
    )
    with engine.begin() as conn:
        conn.execute(
            query,
            {
                "run_id": run_id,
                "source_name": source_name,
                "status": status,
                "records_extracted": records_extracted,
                "file_path": str(file_path) if file_path else None,
                "file_size_bytes": file_size_bytes,
                "error_message": error_message,
            },
        )
    logger.info(f"Ingestion event logged: run_id={run_id}, source={source_name}, status={status}, rows={records_extracted}")


def bulk_copy_df(
    df: pd.DataFrame,
    schema_name: str,
    table_name: str,
    truncate_first: bool = False,
    run_id: Optional[str] = None,
    source_name: Optional[str] = None,
) -> int:
    """
    High-speed bulk copy of a pandas DataFrame to PostgreSQL using psycopg2 COPY FROM STDIN (copy_expert).
    Significantly faster than standard SQL INSERT statements (10x-50x speedup).
    """
    if df.empty:
        logger.warning(f"DataFrame for {schema_name}.{table_name} is empty. Skipping bulk copy.")
        return 0

    if run_id is None:
        run_id = f"load_{int(time.time())}_{uuid.uuid4().hex[:8]}"

    target_table = f"{schema_name}.{table_name}"
    start_time = time.time()
    row_count = len(df)

    logger.info(f"Starting bulk copy of {row_count:,} rows into {target_table} (truncate={truncate_first})")

    # Format column names to lowercase and strip whitespace
    clean_df = df.copy()
    clean_df.columns = [c.strip().lower() for c in clean_df.columns]

    # Query DB to find integer columns and cast to nullable 'Int64' to avoid float string '0.0'
    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            int_cols_query = text(
                """
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_schema = :s AND table_name = :t 
                  AND data_type IN ('smallint', 'integer', 'bigint');
                """
            )
            int_cols = {r[0].lower() for r in conn.execute(int_cols_query, {"s": schema_name, "t": table_name}).fetchall()}

        for col in clean_df.columns:
            if col in int_cols:
                clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce").astype("Int64")
    except Exception as inspect_err:
        logger.debug(f"Could not auto-cast integer columns for {target_table}: {inspect_err}")

    # Convert DataFrame to in-memory CSV buffer
    buffer = io.StringIO()
    clean_df.to_csv(buffer, index=False, header=True, sep="\t", na_rep="\\N")
    buffer.seek(0)

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                if truncate_first:
                    cur.execute(f"TRUNCATE TABLE {target_table} CASCADE;")
                    logger.info(f"Truncated table {target_table} prior to reload.")

                columns_str = ", ".join([f'"{col}"' for col in clean_df.columns])
                copy_sql = f"COPY {target_table} ({columns_str}) FROM STDIN WITH (FORMAT CSV, HEADER TRUE, DELIMITER E'\\t', NULL '\\N');"
                cur.copy_expert(sql=copy_sql, file=buffer)

        duration = time.time() - start_time
        logger.info(f"Successfully loaded {row_count:,} rows into {target_table} in {duration:.2f}s ({row_count/duration:,.0f} rows/s)")

        if source_name:
            log_ingestion_event(
                run_id=run_id,
                source_name=source_name,
                status="success",
                records_extracted=row_count,
            )

        return row_count

    except Exception as e:
        logger.error(f"Failed bulk copy into {target_table}: {e}", exc_info=True)
        if source_name:
            log_ingestion_event(
                run_id=run_id,
                source_name=source_name,
                status="failed",
                records_extracted=0,
                error_message=str(e),
            )
        raise


def get_table_row_count(schema_name: str, table_name: str) -> int:
    """Return the total number of rows in the specified table."""
    engine = get_db_engine()
    with engine.connect() as conn:
        count = conn.execute(text(f"SELECT COUNT(*) FROM {schema_name}.{table_name};")).scalar()
    return count or 0
