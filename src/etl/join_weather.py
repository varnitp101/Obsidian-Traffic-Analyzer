"""
Module: join_weather.py
Description: Spatial and temporal join between clean.accident and staging.weather_hourly.
Joins on 0.25° grid_cell_id and accident timestamp floored to the nearest hour.
Computes coverage metrics, logs join telemetry into ops.dq_results, and implements
robust fallback imputation for records without weather station coverage.
"""

import sys
import time
import json
import logging
from typing import Dict, Tuple
import pandas as pd
from sqlalchemy import text

from src.db.connection import get_db_engine
from src.etl.validate import log_dq_result

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("etl.join_weather")


def evaluate_weather_join_coverage(run_id: str) -> Dict[str, any]:
    """
    Evaluates spatial-temporal matching statistics between clean.accident and staging.weather_hourly.
    Records evaluation metrics in ops.dq_results.
    """
    engine = get_db_engine()
    query = text(
        """
        SELECT 
            COUNT(a.accident_id) AS total_accidents,
            COUNT(w.weather_timestamp) AS matched_accidents,
            COUNT(a.accident_id) - COUNT(w.weather_timestamp) AS unmatched_accidents,
            ROUND(COUNT(w.weather_timestamp)::numeric / COUNT(a.accident_id) * 100, 2) AS match_pct,
            COUNT(DISTINCT a.grid_cell_id) AS total_accident_cells,
            COUNT(DISTINCT w.grid_cell_id) AS covered_weather_cells
        FROM clean.accident a
        LEFT JOIN staging.weather_hourly w
          ON a.grid_cell_id = w.grid_cell_id
         AND DATE_TRUNC('hour', a.accident_ts) = w.weather_timestamp;
        """
    )

    with engine.connect() as conn:
        res = conn.execute(query).fetchone()

    stats = {
        "total_accidents": int(res[0]),
        "matched_accidents": int(res[1]),
        "unmatched_accidents": int(res[2]),
        "match_percentage": float(res[3]),
        "total_accident_cells": int(res[4]),
        "covered_weather_cells": int(res[5]),
    }

    logger.info(
        f"Weather Join Evaluation: Total={stats['total_accidents']:,}, "
        f"Matched={stats['matched_accidents']:,} ({stats['match_percentage']}%), "
        f"Unmatched={stats['unmatched_accidents']:,}"
    )

    log_dq_result(
        run_id=run_id,
        layer_name="clean_to_mart",
        table_name="weather_spatial_join",
        rows_in=stats["total_accidents"],
        rows_passed=stats["matched_accidents"],
        rows_rejected=stats["unmatched_accidents"],
        details=stats
    )

    return stats


def get_accidents_with_weather_query() -> str:
    """
    Returns SQL query that performs the spatial-temporal join between accidents and weather,
    with monthly baseline imputation for accidents outside ingested weather cells.
    """
    return """
        WITH monthly_weather_baseline AS (
            -- Compute monthly UK baselines from observed weather data
            SELECT 
                EXTRACT(MONTH FROM weather_timestamp) AS month_num,
                ROUND(AVG(temperature_2m)::numeric, 1) AS avg_temp,
                ROUND(AVG(precipitation)::numeric, 2) AS avg_precip,
                ROUND(AVG(wind_speed_10m)::numeric, 1) AS avg_wind
            FROM staging.weather_hourly
            GROUP BY 1
        )
        SELECT 
            a.accident_id,
            a.accident_ts,
            EXTRACT(HOUR FROM a.accident_ts)::smallint AS hour,
            a.day_of_week AS weekday,
            EXTRACT(MONTH FROM a.accident_ts)::smallint AS month,
            (a.day_of_week IN (1, 7)) AS is_weekend,
            (EXTRACT(HOUR FROM a.accident_ts) BETWEEN 7 AND 9 OR EXTRACT(HOUR FROM a.accident_ts) BETWEEN 16 AND 18) AS is_rush_hour,
            
            -- Road Features
            COALESCE(a.road_type_label, 'Single carriageway') AS road_type,
            COALESCE(a.speed_limit, 30)::smallint AS speed_limit,
            COALESCE(a.junction_detail_label, 'Not at junction') AS junction_detail,
            COALESCE(a.road_surface_label, 'Dry') AS road_surface,
            COALESCE(a.light_label, 'Daylight') AS light_condition,
            COALESCE(a.weather_label, 'Fine no high winds') AS weather_condition,
            COALESCE(a.urban_rural_label, 'Urban') AS urban_or_rural,
            
            -- Spatial & Context Features
            a.grid_cell_id,
            a.latitude,
            a.longitude,
            COALESCE(a.police_force_code, 0) AS police_force_code,
            COALESCE(a.local_authority_code, 'Unknown') AS local_authority,
            a.num_vehicles,
            a.num_casualties,
            
            -- Joined Weather with Baseline Fallback Imputation
            COALESCE(w.temperature_2m, b.avg_temp, 11.5) AS temperature_2m,
            COALESCE(w.precipitation, b.avg_precip, 0.0) AS precipitation,
            COALESCE(w.wind_speed_10m, b.avg_wind, 14.2) AS wind_speed_10m,
            
            -- Target
            a.severity_code,
            a.severity_label
            
        FROM clean.accident a
        LEFT JOIN staging.weather_hourly w
          ON a.grid_cell_id = w.grid_cell_id
         AND DATE_TRUNC('hour', a.accident_ts) = w.weather_timestamp
        LEFT JOIN monthly_weather_baseline b
          ON EXTRACT(MONTH FROM a.accident_ts) = b.month_num;
    """


def fetch_joined_accidents(limit: int = None) -> pd.DataFrame:
    """
    Fetch accidents joined with historical weather data into a pandas DataFrame.
    """
    engine = get_db_engine()
    sql = get_accidents_with_weather_query()
    if limit:
        sql += f" LIMIT {limit}"

    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn)

    logger.info(f"Retrieved {len(df):,} accident records with joined weather features.")
    return df


if __name__ == "__main__":
    run_id = f"eval_join_{int(time.time())}"
    stats = evaluate_weather_join_coverage(run_id)
    print("\n--- Weather Spatial Join Evaluation ---")
    for k, v in stats.items():
        print(f"  {k}: {v}")
