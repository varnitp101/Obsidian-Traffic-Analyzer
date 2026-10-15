"""
Module: clean.py
Description: Data transformation, standardization, lookup code decoding,
and loading into PostgreSQL clean schema (clean.accident, clean.vehicle, clean.casualty).
"""

import sys
import time
import uuid
import logging
from typing import Optional, Dict
import pandas as pd
from sqlalchemy import text

from src.db.connection import get_db_engine
from src.db.load import bulk_copy_df
from src.etl.validate import (
    validate_collisions,
    validate_vehicles,
    validate_casualties,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("etl.clean")

# -----------------------------------------------------------------------------
# STATS19 Official Lookup Code Dictionaries
# -----------------------------------------------------------------------------
SEVERITY_LOOKUP = {
    1: "Fatal",
    2: "Serious",
    3: "Slight",
}

ROAD_TYPE_LOOKUP = {
    1: "Roundabout",
    2: "One way street",
    3: "Dual carriageway",
    6: "Single carriageway",
    7: "Slip road",
    9: "Unknown",
}

LIGHT_LOOKUP = {
    1: "Daylight",
    4: "Darkness - lights lit",
    5: "Darkness - lights unlit",
    6: "Darkness - no lighting",
    7: "Darkness - lighting unknown",
}

WEATHER_LOOKUP = {
    1: "Fine no high winds",
    2: "Raining no high winds",
    3: "Snowing no high winds",
    4: "Fine + high winds",
    5: "Raining + high winds",
    6: "Snowing + high winds",
    7: "Fog or mist",
    8: "Other",
    9: "Unknown",
}

ROAD_SURFACE_LOOKUP = {
    1: "Dry",
    2: "Wet or damp",
    3: "Snow",
    4: "Frost or ice",
    5: "Flood over 3cm",
    9: "Unknown",
}

URBAN_RURAL_LOOKUP = {
    1: "Urban",
    2: "Rural",
    3: "Unallocated",
}

JUNCTION_DETAIL_LOOKUP = {
    0: "Not at junction",
    1: "Roundabout",
    2: "Mini-roundabout",
    3: "T or staggered junction",
    5: "Slip road",
    6: "Crossroads",
    7: "More than 4 arms",
    8: "Private drive or entrance",
    9: "Other junction",
}

SEX_LOOKUP = {
    1: "Male",
    2: "Female",
    3: "Not known",
}

CASUALTY_CLASS_LOOKUP = {
    1: "Driver or rider",
    2: "Passenger",
    3: "Pedestrian",
}

VEHICLE_TYPE_LOOKUP = {
    1: "Pedal cycle",
    2: "Motorcycle <=50cc",
    3: "Motorcycle <=125cc",
    4: "Motorcycle 125-500cc",
    5: "Motorcycle >500cc",
    8: "Taxi/Private hire",
    9: "Car",
    10: "Minibus",
    11: "Bus or coach",
    19: "Van / Goods <=3.5t",
    20: "Goods 3.5t-7.5t",
    21: "Goods >=7.5t",
    90: "Other vehicle",
}


def safe_int(val, invalid_values=(-1, 99, 999, -99)) -> Optional[int]:
    """Safely cast value to int or return None if null or sentinel."""
    if pd.isna(val):
        return None
    try:
        i = int(float(val))
        return None if i in invalid_values else i
    except (ValueError, TypeError):
        return None


def clean_collisions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize validated collisions to match clean.accident schema:
    - Combines date & time into accident_ts
    - Computes spatial grid_cell_id (0.25° resolution)
    - Decodes categorical integers into human-readable labels
    """
    logger.info(f"Standardizing and decoding {len(df):,} validated collision records...")
    id_col = "collision_index" if "collision_index" in df.columns else "accident_index"

    records = []
    for _, row in df.iterrows():
        # Parse timestamp: Date is DD/MM/YYYY
        date_str = str(row["date"]).strip()
        time_str = str(row.get("time", "00:00")).strip()
        if not time_str or time_str == "nan":
            time_str = "00:00"
        if len(time_str) == 4 and ":" not in time_str:
            time_str = f"{time_str[:2]}:{time_str[2:]}"
        elif len(time_str) == 5:
            pass
        else:
            time_str = "00:00"

        # Form YYYY-MM-DD HH:MM:00
        d_parts = date_str.split("/")
        iso_ts = f"{d_parts[2]}-{d_parts[1].zfill(2)}-{d_parts[0].zfill(2)} {time_str}:00"

        lat = float(row["latitude"])
        lon = float(row["longitude"])
        grid_lat = round(lat / 0.25) * 0.25
        grid_lon = round(lon / 0.25) * 0.25
        grid_cell_id = f"GRID_{grid_lat:.2f}_{grid_lon:.2f}"

        sev_code = int(row.get("collision_severity", row.get("accident_severity", 3)))
        road_type_c = safe_int(row.get("road_type"))
        light_c = safe_int(row.get("light_conditions"))
        weather_c = safe_int(row.get("weather_conditions"))
        surface_c = safe_int(row.get("road_surface_conditions"))
        urban_rural_c = safe_int(row.get("urban_or_rural_area"))
        junc_detail_c = safe_int(row.get("junction_detail"))
        speed_lim = safe_int(row.get("speed_limit"))
        police_force_c = safe_int(row.get("police_force"))

        records.append({
            "accident_id": str(row[id_col]),
            "accident_ts": iso_ts,
            "latitude": lat,
            "longitude": lon,
            "grid_cell_id": grid_cell_id,
            "severity_code": sev_code,
            "severity_label": SEVERITY_LOOKUP.get(sev_code, "Unknown"),
            "num_vehicles": int(row.get("number_of_vehicles", 1)),
            "num_casualties": int(row.get("number_of_casualties", 1)),
            "day_of_week": int(row.get("day_of_week", 1)),
            "road_type_code": road_type_c,
            "road_type_label": ROAD_TYPE_LOOKUP.get(road_type_c, "Unknown") if road_type_c else None,
            "speed_limit": speed_lim,
            "junction_detail_code": junc_detail_c,
            "junction_detail_label": JUNCTION_DETAIL_LOOKUP.get(junc_detail_c, "Unknown") if junc_detail_c is not None else None,
            "light_code": light_c,
            "light_label": LIGHT_LOOKUP.get(light_c, "Unknown") if light_c else None,
            "weather_code": weather_c,
            "weather_label": WEATHER_LOOKUP.get(weather_c, "Unknown") if weather_c else None,
            "road_surface_code": surface_c,
            "road_surface_label": ROAD_SURFACE_LOOKUP.get(surface_c, "Unknown") if surface_c else None,
            "urban_rural_code": urban_rural_c,
            "urban_rural_label": URBAN_RURAL_LOOKUP.get(urban_rural_c, "Unknown") if urban_rural_c else None,
            "police_force_code": police_force_c,
            "local_authority_code": str(row.get("local_authority_district", "")) if not pd.isna(row.get("local_authority_district")) else None,
        })

    return pd.DataFrame(records)


def clean_vehicles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize validated vehicles to match clean.vehicle schema.
    """
    logger.info(f"Standardizing and decoding {len(df):,} validated vehicle records...")
    id_col = "collision_index" if "collision_index" in df.columns else "accident_index"

    records = []
    for _, row in df.iterrows():
        v_type_c = safe_int(row.get("vehicle_type"))
        manoeuvre_c = safe_int(row.get("vehicle_manoeuvre"))
        sex_c = safe_int(row.get("sex_of_driver"))
        age = safe_int(row.get("age_of_driver"))
        age_band = safe_int(row.get("age_band_of_driver"))
        engine_cc = safe_int(row.get("engine_capacity_cc"))
        veh_age = safe_int(row.get("age_of_vehicle"))

        records.append({
            "accident_id": str(row[id_col]),
            "vehicle_ref": int(row["vehicle_reference"]),
            "vehicle_type_code": v_type_c,
            "vehicle_type_label": VEHICLE_TYPE_LOOKUP.get(v_type_c, "Other") if v_type_c else None,
            "manoeuvre_code": manoeuvre_c,
            "manoeuvre_label": str(row.get("vehicle_manoeuvre", "")) if manoeuvre_c else None,
            "sex_of_driver_code": sex_c,
            "sex_of_driver_label": SEX_LOOKUP.get(sex_c, "Not known") if sex_c else None,
            "age_of_driver": age,
            "age_band_driver_code": age_band,
            "engine_capacity_cc": engine_cc,
            "age_of_vehicle": veh_age,
        })

    return pd.DataFrame(records)


def clean_casualties(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize validated casualties to match clean.casualty schema.
    """
    logger.info(f"Standardizing and decoding {len(df):,} validated casualty records...")
    id_col = "collision_index" if "collision_index" in df.columns else "accident_index"

    records = []
    for _, row in df.iterrows():
        v_ref = safe_int(row.get("vehicle_reference"))
        c_class_c = safe_int(row.get("casualty_class"))
        sex_c = safe_int(row.get("sex_of_casualty"))
        age = safe_int(row.get("age_of_casualty"))
        sev_c = safe_int(row.get("casualty_severity"))

        records.append({
            "accident_id": str(row[id_col]),
            "vehicle_ref": v_ref,
            "casualty_ref": int(row["casualty_reference"]),
            "casualty_class_code": c_class_c,
            "casualty_class_label": CASUALTY_CLASS_LOOKUP.get(c_class_c, "Unknown") if c_class_c else None,
            "sex_of_casualty_code": sex_c,
            "sex_of_casualty_label": SEX_LOOKUP.get(sex_c, "Not known") if sex_c else None,
            "age_of_casualty": age,
            "casualty_severity_code": sev_c,
            "casualty_severity_label": SEVERITY_LOOKUP.get(sev_c, "Unknown") if sev_c else None,
        })

    return pd.DataFrame(records)


def run_clean_and_validate(truncate_clean: bool = True) -> Dict[str, int]:
    """
    End-to-end cleaning and validation workflow:
    1. Reads staging tables.
    2. Runs validation rules and negative logging to ops.rejected_records and data/rejected/.
    3. Transforms and decodes validated rows.
    4. Bulk loads into clean.accident, clean.vehicle, clean.casualty.
    """
    run_id = f"clean_run_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    engine = get_db_engine()
    start_time = time.time()

    logger.info(f"=== Starting Cleaning & Validation Pipeline (run_id={run_id}) ===")

    # 1. Collisions
    with engine.connect() as conn:
        raw_collisions = pd.read_sql("SELECT * FROM staging.collision;", conn)
    logger.info(f"Loaded {len(raw_collisions):,} raw collisions from staging.")

    valid_collisions, rejected_collisions = validate_collisions(raw_collisions, run_id=run_id)
    cleaned_accidents = clean_collisions(valid_collisions)

    # Bulk load into clean.accident
    bulk_copy_df(
        df=cleaned_accidents,
        schema_name="clean",
        table_name="accident",
        truncate_first=truncate_clean,
        run_id=f"{run_id}_accidents",
        source_name="clean_accident"
    )

    valid_collision_ids = set(cleaned_accidents["accident_id"].tolist())
    logger.info(f"Verified {len(valid_collision_ids):,} valid unique accident IDs for foreign key checks.")

    # 2. Vehicles
    with engine.connect() as conn:
        raw_vehicles = pd.read_sql("SELECT * FROM staging.vehicle;", conn)
    logger.info(f"Loaded {len(raw_vehicles):,} raw vehicles from staging.")

    valid_vehicles, rejected_vehicles = validate_vehicles(raw_vehicles, valid_collision_ids, run_id=run_id)
    cleaned_vehicles = clean_vehicles(valid_vehicles)

    bulk_copy_df(
        df=cleaned_vehicles,
        schema_name="clean",
        table_name="vehicle",
        truncate_first=truncate_clean,
        run_id=f"{run_id}_vehicles",
        source_name="clean_vehicle"
    )

    # 3. Casualties
    with engine.connect() as conn:
        raw_casualties = pd.read_sql("SELECT * FROM staging.casualty;", conn)
    logger.info(f"Loaded {len(raw_casualties):,} raw casualties from staging.")

    valid_casualties, rejected_casualties = validate_casualties(raw_casualties, valid_collision_ids, run_id=run_id)
    cleaned_casualties = clean_casualties(valid_casualties)

    bulk_copy_df(
        df=cleaned_casualties,
        schema_name="clean",
        table_name="casualty",
        truncate_first=truncate_clean,
        run_id=f"{run_id}_casualties",
        source_name="clean_casualty"
    )

    elapsed = time.time() - start_time
    summary = {
        "accidents_in": len(raw_collisions),
        "accidents_passed": len(cleaned_accidents),
        "accidents_rejected": len(rejected_collisions),
        "vehicles_in": len(raw_vehicles),
        "vehicles_passed": len(cleaned_vehicles),
        "vehicles_rejected": len(rejected_vehicles),
        "casualties_in": len(raw_casualties),
        "casualties_passed": len(cleaned_casualties),
        "casualties_rejected": len(rejected_casualties),
        "duration_seconds": round(elapsed, 2)
    }

    logger.info(f"=== Cleaning & Validation Pipeline Complete in {elapsed:.2f}s ===")
    logger.info(f"Summary: {summary}")
    return summary


if __name__ == "__main__":
    res = run_clean_and_validate(truncate_clean=True)
    print("\n--- Cleaning & Validation Execution Summary ---")
    for k, v in res.items():
        print(f"  {k}: {v}")
