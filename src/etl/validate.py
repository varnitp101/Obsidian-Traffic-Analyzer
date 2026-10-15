"""
Module: validate.py
Description: Data quality rules, negative validation filters, and rejected-record logging.
Implements rule-by-rule evaluation for collisions, vehicles, and casualties.
Failing records are logged to ops.rejected_records and saved to data/rejected/*.csv.
Summary metrics are recorded in ops.dq_results.
"""

import os
import sys
import json
import time
import uuid
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
import pandas as pd
from sqlalchemy import text

from src.db.connection import get_db_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("etl.validate")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REJECTED_DIR = REPO_ROOT / "data" / "rejected"


def log_rejected_records(
    rejections: List[Dict],
    run_id: str,
    source_table: str,
    save_csv: bool = True
) -> int:
    """
    Log rejected records to PostgreSQL ops.rejected_records and save to data/rejected/*.csv.
    """
    if not rejections:
        logger.info(f"No rejected records to log for {source_table}.")
        return 0

    count = len(rejections)
    logger.warning(f"Logging {count:,} rejected records for {source_table} (run_id={run_id})...")

    # 1. Save to CSV in data/rejected/
    if save_csv:
        REJECTED_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = REJECTED_DIR / f"rejected_{source_table}_{run_id}.csv"
        df_rej = pd.DataFrame(rejections)
        df_rej.to_csv(csv_path, index=False)
        logger.info(f"Saved rejected records CSV to {csv_path}")

    # 2. Insert into ops.rejected_records in batches
    engine = get_db_engine()
    insert_sql = text(
        """
        INSERT INTO ops.rejected_records (
            run_id, source_table, record_id, rule_name, reason, raw_payload, rejected_at
        )
        VALUES (
            :run_id, :source_table, :record_id, :rule_name, :reason, :raw_payload, CURRENT_TIMESTAMP
        );
        """
    )

    batch_params = []
    for r in rejections:
        payload = r.get("raw_payload", {})
        if isinstance(payload, dict):
            # Replace NaNs / pandas NAs with None so JSON serialization produces valid 'null'
            sanitized = {k: (None if pd.isna(v) else v) for k, v in payload.items()}
            payload_json = json.dumps(sanitized)
        else:
            payload_json = json.dumps(str(payload))

        batch_params.append({
            "run_id": run_id,
            "source_table": source_table,
            "record_id": str(r.get("record_id", "UNKNOWN")),
            "rule_name": str(r.get("rule_name", "UNKNOWN_RULE")),
            "reason": str(r.get("reason", "Validation failed")),
            "raw_payload": payload_json,
        })

    with engine.begin() as conn:
        conn.execute(insert_sql, batch_params)

    logger.info(f"Successfully recorded {count:,} rejections in ops.rejected_records.")
    return count


def log_dq_result(
    run_id: str,
    layer_name: str,
    table_name: str,
    rows_in: int,
    rows_passed: int,
    rows_rejected: int,
    details: Dict
) -> None:
    """
    Record data quality evaluation summary metrics into ops.dq_results.
    """
    engine = get_db_engine()
    query = text(
        """
        INSERT INTO ops.dq_results (
            run_id, layer_name, table_name, rows_in, rows_passed, rows_rejected, dq_metric_details, evaluated_at
        )
        VALUES (
            :run_id, :layer_name, :table_name, :rows_in, :rows_passed, :rows_rejected, :dq_metric_details, CURRENT_TIMESTAMP
        );
        """
    )
    with engine.begin() as conn:
        conn.execute(
            query,
            {
                "run_id": run_id,
                "layer_name": layer_name,
                "table_name": table_name,
                "rows_in": rows_in,
                "rows_passed": rows_passed,
                "rows_rejected": rows_rejected,
                "dq_metric_details": json.dumps(details),
            }
        )
    logger.info(
        f"DQ Result logged: {layer_name}.{table_name} - In={rows_in:,}, Passed={rows_passed:,}, Rejected={rows_rejected:,}"
    )


# -----------------------------------------------------------------------------
# Validation Rules for Collisions
# -----------------------------------------------------------------------------
def validate_collisions(
    df: pd.DataFrame,
    run_id: str,
    uk_bounds: Tuple[float, float, float, float] = (49.0, 61.0, -9.0, 2.0)
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Applies data quality and validity rules to collision records:
    1. Duplicate accident_index (keeps first, rejects duplicates).
    2. Missing or null latitude/longitude.
    3. Coordinates outside UK bounding box or exactly (0, 0).
    4. Missing or unparseable date/time.
    5. Severity code not in {1, 2, 3}.
    6. Number of vehicles or casualties < 1.
    """
    rows_in = len(df)
    min_lat, max_lat, min_lon, max_lon = uk_bounds
    rejections = []
    rejected_indices = set()
    rule_counts = {}

    def add_rejection(idx, record_id, rule, reason, row_dict):
        if idx not in rejected_indices:
            rejected_indices.add(idx)
            rejections.append({
                "record_id": record_id,
                "rule_name": rule,
                "reason": reason,
                "raw_payload": row_dict
            })
            rule_counts[rule] = rule_counts.get(rule, 0) + 1

    # Rule 1: Duplicate collision_index
    id_col = "collision_index" if "collision_index" in df.columns else "accident_index"
    duplicated_mask = df.duplicated(subset=[id_col], keep="first")
    for idx in df[duplicated_mask].index:
        row = df.loc[idx]
        add_rejection(idx, row[id_col], "DUPLICATE_ID", f"Duplicate collision index: {row[id_col]}", row.to_dict())

    # Rule 2 & 3: Coordinate validity
    for idx, row in df.iterrows():
        if idx in rejected_indices:
            continue
        rec_id = row[id_col]

        lat_val, lon_val = row.get("latitude"), row.get("longitude")
        if pd.isna(lat_val) or pd.isna(lon_val) or str(lat_val).strip() == "" or str(lon_val).strip() == "":
            add_rejection(idx, rec_id, "NULL_COORDINATES", "Latitude or longitude is null or empty", row.to_dict())
            continue

        try:
            lat = float(lat_val)
            lon = float(lon_val)
        except (ValueError, TypeError):
            add_rejection(idx, rec_id, "INVALID_COORDINATES", f"Non-numeric coordinates: lat={lat_val}, lon={lon_val}", row.to_dict())
            continue

        if (lat == 0.0 and lon == 0.0) or not (min_lat <= lat <= max_lat and min_lon <= lon <= max_lon):
            add_rejection(idx, rec_id, "OUT_OF_BOUNDS_COORDINATES", f"Coordinates ({lat:.4f}, {lon:.4f}) outside UK bounding box", row.to_dict())
            continue

        # Rule 4: Date & Time validity
        date_val = str(row.get("date", "")).strip()
        time_val = str(row.get("time", "")).strip()
        if not date_val:
            add_rejection(idx, rec_id, "MISSING_DATE", "Date field is missing or empty", row.to_dict())
            continue

        # Validate date format DD/MM/YYYY
        date_parts = date_val.split("/")
        if len(date_parts) != 3 or len(date_parts[2]) != 4:
            add_rejection(idx, rec_id, "INVALID_DATE_FORMAT", f"Unparseable date: {date_val}", row.to_dict())
            continue

        # Rule 5: Severity code in {1, 2, 3}
        sev_val = row.get("collision_severity", row.get("accident_severity"))
        try:
            sev_int = int(sev_val)
            if sev_int not in (1, 2, 3):
                add_rejection(idx, rec_id, "INVALID_SEVERITY_CODE", f"Severity code {sev_int} not in (1, 2, 3)", row.to_dict())
                continue
        except (ValueError, TypeError):
            add_rejection(idx, rec_id, "NON_INTEGER_SEVERITY", f"Unparseable severity code: {sev_val}", row.to_dict())
            continue

        # Rule 6: Absurd or negative counts
        try:
            n_veh = int(row.get("number_of_vehicles", 1))
            n_cas = int(row.get("number_of_casualties", 1))
            if n_veh < 1 or n_cas < 1:
                add_rejection(idx, rec_id, "ABSURD_COUNTS", f"Vehicles={n_veh}, Casualties={n_cas} must be >= 1", row.to_dict())
                continue
        except (ValueError, TypeError):
            add_rejection(idx, rec_id, "INVALID_COUNTS", f"Unparseable vehicle or casualty counts", row.to_dict())
            continue

    passed_df = df.drop(index=list(rejected_indices)).copy()
    rejected_df = pd.DataFrame(rejections)

    # Log telemetry
    log_rejected_records(rejections, run_id=run_id, source_table="collision")
    log_dq_result(
        run_id=run_id,
        layer_name="clean",
        table_name="accident",
        rows_in=rows_in,
        rows_passed=len(passed_df),
        rows_rejected=len(rejections),
        details={"rule_breakdown": rule_counts}
    )

    return passed_df, rejected_df


# -----------------------------------------------------------------------------
# Validation Rules for Vehicles
# -----------------------------------------------------------------------------
def validate_vehicles(
    df: pd.DataFrame,
    valid_collision_ids: Set[str],
    run_id: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validates vehicles:
    1. Orphan check (collision_index must exist in validated collisions).
    2. Vehicle reference must be positive integer.
    """
    rows_in = len(df)
    rejections = []
    rejected_indices = set()
    rule_counts = {}

    id_col = "collision_index" if "collision_index" in df.columns else "accident_index"

    for idx, row in df.iterrows():
        rec_id = f"{row[id_col]}_{row.get('vehicle_reference', '0')}"

        # Rule: Orphan Check
        if row[id_col] not in valid_collision_ids:
            rejected_indices.add(idx)
            rejections.append({
                "record_id": rec_id,
                "rule_name": "ORPHAN_VEHICLE",
                "reason": f"Collision index {row[id_col]} does not exist in valid collisions",
                "raw_payload": row.to_dict()
            })
            rule_counts["ORPHAN_VEHICLE"] = rule_counts.get("ORPHAN_VEHICLE", 0) + 1
            continue

        # Rule: Valid vehicle reference
        try:
            v_ref = int(row.get("vehicle_reference", -1))
            if v_ref < 1:
                rejected_indices.add(idx)
                rejections.append({
                    "record_id": rec_id,
                    "rule_name": "INVALID_VEHICLE_REF",
                    "reason": f"Vehicle reference {v_ref} must be >= 1",
                    "raw_payload": row.to_dict()
                })
                rule_counts["INVALID_VEHICLE_REF"] = rule_counts.get("INVALID_VEHICLE_REF", 0) + 1
                continue
        except (ValueError, TypeError):
            rejected_indices.add(idx)
            rejections.append({
                "record_id": rec_id,
                "rule_name": "NON_NUMERIC_VEHICLE_REF",
                "reason": f"Unparseable vehicle reference",
                "raw_payload": row.to_dict()
            })
            rule_counts["NON_NUMERIC_VEHICLE_REF"] = rule_counts.get("NON_NUMERIC_VEHICLE_REF", 0) + 1
            continue

    passed_df = df.drop(index=list(rejected_indices)).copy()
    rejected_df = pd.DataFrame(rejections)

    log_rejected_records(rejections, run_id=run_id, source_table="vehicle")
    log_dq_result(
        run_id=run_id,
        layer_name="clean",
        table_name="vehicle",
        rows_in=rows_in,
        rows_passed=len(passed_df),
        rows_rejected=len(rejections),
        details={"rule_breakdown": rule_counts}
    )

    return passed_df, rejected_df


# -----------------------------------------------------------------------------
# Validation Rules for Casualties
# -----------------------------------------------------------------------------
def validate_casualties(
    df: pd.DataFrame,
    valid_collision_ids: Set[str],
    run_id: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validates casualties:
    1. Orphan check (collision_index must exist in validated collisions).
    2. Casualty reference must be positive integer.
    3. Severity code in {1, 2, 3}.
    """
    rows_in = len(df)
    rejections = []
    rejected_indices = set()
    rule_counts = {}

    id_col = "collision_index" if "collision_index" in df.columns else "accident_index"

    for idx, row in df.iterrows():
        rec_id = f"{row[id_col]}_{row.get('casualty_reference', '0')}"

        # Rule: Orphan Check
        if row[id_col] not in valid_collision_ids:
            rejected_indices.add(idx)
            rejections.append({
                "record_id": rec_id,
                "rule_name": "ORPHAN_CASUALTY",
                "reason": f"Collision index {row[id_col]} does not exist in valid collisions",
                "raw_payload": row.to_dict()
            })
            rule_counts["ORPHAN_CASUALTY"] = rule_counts.get("ORPHAN_CASUALTY", 0) + 1
            continue

        # Rule: Valid casualty reference
        try:
            c_ref = int(row.get("casualty_reference", -1))
            if c_ref < 1:
                rejected_indices.add(idx)
                rejections.append({
                    "record_id": rec_id,
                    "rule_name": "INVALID_CASUALTY_REF",
                    "reason": f"Casualty reference {c_ref} must be >= 1",
                    "raw_payload": row.to_dict()
                })
                rule_counts["INVALID_CASUALTY_REF"] = rule_counts.get("INVALID_CASUALTY_REF", 0) + 1
                continue
        except (ValueError, TypeError):
            rejected_indices.add(idx)
            rejections.append({
                "record_id": rec_id,
                "rule_name": "NON_NUMERIC_CASUALTY_REF",
                "reason": f"Unparseable casualty reference",
                "raw_payload": row.to_dict()
            })
            rule_counts["NON_NUMERIC_CASUALTY_REF"] = rule_counts.get("NON_NUMERIC_CASUALTY_REF", 0) + 1
            continue

        # Rule: Valid casualty severity code
        sev_val = row.get("casualty_severity")
        try:
            sev_int = int(sev_val)
            if sev_int not in (1, 2, 3):
                rejected_indices.add(idx)
                rejections.append({
                    "record_id": rec_id,
                    "rule_name": "INVALID_CASUALTY_SEVERITY",
                    "reason": f"Casualty severity {sev_int} not in (1, 2, 3)",
                    "raw_payload": row.to_dict()
                })
                rule_counts["INVALID_CASUALTY_SEVERITY"] = rule_counts.get("INVALID_CASUALTY_SEVERITY", 0) + 1
                continue
        except (ValueError, TypeError):
            rejected_indices.add(idx)
            rejections.append({
                "record_id": rec_id,
                "rule_name": "NON_NUMERIC_CASUALTY_SEVERITY",
                "reason": f"Unparseable casualty severity code",
                "raw_payload": row.to_dict()
            })
            rule_counts["NON_NUMERIC_CASUALTY_SEVERITY"] = rule_counts.get("NON_NUMERIC_CASUALTY_SEVERITY", 0) + 1
            continue

    passed_df = df.drop(index=list(rejected_indices)).copy()
    rejected_df = pd.DataFrame(rejections)

    log_rejected_records(rejections, run_id=run_id, source_table="casualty")
    log_dq_result(
        run_id=run_id,
        layer_name="clean",
        table_name="casualty",
        rows_in=rows_in,
        rows_passed=len(passed_df),
        rows_rejected=len(rejections),
        details={"rule_breakdown": rule_counts}
    )

    return passed_df, rejected_df
