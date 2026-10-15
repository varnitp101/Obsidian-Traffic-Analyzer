"""
Module: build_mart.py
Description: Dimensional model (Star Schema) and mart.model_ready generator.
Populates:
  - Dimensions: dim_date, dim_road, dim_weather, dim_location, dim_severity
  - Fact: fact_accident (with surrogate keys and measures)
  - Model-ready: mart.model_ready (wide, leak-free, production ML input table)
"""

import sys
import time
import logging
from typing import Dict
from sqlalchemy import text

from src.db.connection import get_db_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("etl.build_mart")


def build_dim_date() -> int:
    """Populate mart.dim_date covering all unique hourly timestamps."""
    logger.info("Populating mart.dim_date...")
    engine = get_db_engine()
    sql = text(
        """
        INSERT INTO mart.dim_date (
            date_key, full_date, year, month, month_name, day, 
            day_of_week, weekday_name, hour, is_weekend, is_rush_hour
        )
        SELECT DISTINCT
            (TO_CHAR(accident_ts, 'YYYYMMDD')::integer * 100 + EXTRACT(HOUR FROM accident_ts)::integer) AS date_key,
            accident_ts::date AS full_date,
            EXTRACT(YEAR FROM accident_ts)::smallint AS year,
            EXTRACT(MONTH FROM accident_ts)::smallint AS month,
            TO_CHAR(accident_ts, 'Month') AS month_name,
            EXTRACT(DAY FROM accident_ts)::smallint AS day,
            EXTRACT(DOW FROM accident_ts)::smallint AS day_of_week,
            TO_CHAR(accident_ts, 'Day') AS weekday_name,
            EXTRACT(HOUR FROM accident_ts)::smallint AS hour,
            (EXTRACT(DOW FROM accident_ts) IN (0, 6)) AS is_weekend,
            (EXTRACT(HOUR FROM accident_ts) BETWEEN 7 AND 9 OR EXTRACT(HOUR FROM accident_ts) BETWEEN 16 AND 18) AS is_rush_hour
        FROM clean.accident
        ON CONFLICT (date_key) DO NOTHING;
        """
    )
    with engine.begin() as conn:
        res = conn.execute(sql)
        count = conn.execute(text("SELECT COUNT(*) FROM mart.dim_date;")).scalar()
    logger.info(f"mart.dim_date populated with {count:,} hourly dimension records.")
    return count


def build_dim_road() -> int:
    """Populate mart.dim_road with unique road condition attributes."""
    logger.info("Populating mart.dim_road...")
    engine = get_db_engine()
    sql = text(
        """
        INSERT INTO mart.dim_road (road_type, speed_limit, junction_detail, road_surface)
        SELECT DISTINCT
            COALESCE(road_type_label, 'Single carriageway') AS road_type,
            COALESCE(speed_limit, 30)::smallint AS speed_limit,
            COALESCE(junction_detail_label, 'Not at junction') AS junction_detail,
            COALESCE(road_surface_label, 'Dry') AS road_surface
        FROM clean.accident
        ON CONFLICT (road_type, speed_limit, junction_detail, road_surface) DO NOTHING;
        """
    )
    with engine.begin() as conn:
        conn.execute(sql)
        count = conn.execute(text("SELECT COUNT(*) FROM mart.dim_road;")).scalar()
    logger.info(f"mart.dim_road populated with {count:,} unique road configurations.")
    return count


def build_dim_weather() -> int:
    """Populate mart.dim_weather with categorized atmospheric combinations."""
    logger.info("Populating mart.dim_weather...")
    engine = get_db_engine()
    sql = text(
        """
        WITH categorized_weather AS (
            SELECT DISTINCT
                COALESCE(a.weather_label, 'Fine no high winds') AS weather_condition,
                COALESCE(a.light_label, 'Daylight') AS light_condition,
                CASE 
                    WHEN COALESCE(w.temperature_2m, 11.5) < 0 THEN 'Freezing'
                    WHEN COALESCE(w.temperature_2m, 11.5) < 10 THEN 'Cold'
                    WHEN COALESCE(w.temperature_2m, 11.5) < 18 THEN 'Mild'
                    WHEN COALESCE(w.temperature_2m, 11.5) < 25 THEN 'Warm'
                    ELSE 'Hot'
                END AS temp_category,
                CASE 
                    WHEN COALESCE(w.precipitation, 0.0) = 0 THEN 'None'
                    WHEN COALESCE(w.precipitation, 0.0) <= 2.5 THEN 'Light'
                    WHEN COALESCE(w.precipitation, 0.0) <= 7.5 THEN 'Moderate'
                    ELSE 'Heavy'
                END AS rain_category,
                CASE 
                    WHEN COALESCE(w.wind_speed_10m, 14.2) < 10 THEN 'Calm'
                    WHEN COALESCE(w.wind_speed_10m, 14.2) <= 30 THEN 'Breeze'
                    ELSE 'Gale'
                END AS wind_category
            FROM clean.accident a
            LEFT JOIN staging.weather_hourly w
              ON a.grid_cell_id = w.grid_cell_id
             AND DATE_TRUNC('hour', a.accident_ts) = w.weather_timestamp
        )
        INSERT INTO mart.dim_weather (weather_condition, light_condition, temp_category, rain_category, wind_category)
        SELECT weather_condition, light_condition, temp_category, rain_category, wind_category
        FROM categorized_weather
        ON CONFLICT (weather_condition, light_condition, temp_category, rain_category, wind_category) DO NOTHING;
        """
    )
    with engine.begin() as conn:
        conn.execute(sql)
        count = conn.execute(text("SELECT COUNT(*) FROM mart.dim_weather;")).scalar()
    logger.info(f"mart.dim_weather populated with {count:,} atmospheric profiles.")
    return count


def build_dim_location() -> int:
    """Populate mart.dim_location with geographic grid and jurisdiction metadata."""
    logger.info("Populating mart.dim_location...")
    engine = get_db_engine()
    sql = text(
        """
        INSERT INTO mart.dim_location (grid_cell_id, grid_latitude, grid_longitude, local_authority, police_force, urban_rural)
        SELECT DISTINCT
            a.grid_cell_id,
            ROUND(a.latitude::numeric / 0.25) * 0.25 AS grid_latitude,
            ROUND(a.longitude::numeric / 0.25) * 0.25 AS grid_longitude,
            COALESCE(a.local_authority_code, 'Unknown') AS local_authority,
            CONCAT('Police Force ', COALESCE(a.police_force_code, 0)) AS police_force,
            COALESCE(a.urban_rural_label, 'Urban') AS urban_rural
        FROM clean.accident a
        ON CONFLICT (grid_cell_id, local_authority, police_force, urban_rural) DO NOTHING;
        """
    )
    with engine.begin() as conn:
        conn.execute(sql)
        count = conn.execute(text("SELECT COUNT(*) FROM mart.dim_location;")).scalar()
    logger.info(f"mart.dim_location populated with {count:,} geographic locations.")
    return count


def build_fact_accident() -> int:
    """Populate mart.fact_accident by joining clean records with dimension surrogate keys."""
    logger.info("Populating mart.fact_accident...")
    engine = get_db_engine()
    sql = text(
        """
        TRUNCATE TABLE mart.fact_accident CASCADE;
        
        WITH monthly_weather_baseline AS (
            SELECT 
                EXTRACT(MONTH FROM weather_timestamp) AS month_num,
                AVG(temperature_2m) AS avg_temp,
                AVG(precipitation) AS avg_precip,
                AVG(wind_speed_10m) AS avg_wind
            FROM staging.weather_hourly
            GROUP BY 1
        )
        INSERT INTO mart.fact_accident (
            accident_id, date_key, road_key, weather_key, location_key, severity_key,
            num_vehicles, num_casualties, temperature_2m, precipitation, wind_speed_10m,
            latitude, longitude
        )
        SELECT 
            a.accident_id,
            d.date_key,
            r.road_key,
            w_dim.weather_key,
            loc.location_key,
            a.severity_code AS severity_key,
            
            a.num_vehicles,
            a.num_casualties,
            COALESCE(w.temperature_2m, b.avg_temp, 11.5) AS temperature_2m,
            COALESCE(w.precipitation, b.avg_precip, 0.0) AS precipitation,
            COALESCE(w.wind_speed_10m, b.avg_wind, 14.2) AS wind_speed_10m,
            a.latitude,
            a.longitude
            
        FROM clean.accident a
        
        -- Dimension Joins
        INNER JOIN mart.dim_date d
          ON d.date_key = (TO_CHAR(a.accident_ts, 'YYYYMMDD')::integer * 100 + EXTRACT(HOUR FROM a.accident_ts)::integer)
          
        INNER JOIN mart.dim_road r
          ON r.road_type = COALESCE(a.road_type_label, 'Single carriageway')
         AND r.speed_limit = COALESCE(a.speed_limit, 30)::smallint
         AND r.junction_detail = COALESCE(a.junction_detail_label, 'Not at junction')
         AND r.road_surface = COALESCE(a.road_surface_label, 'Dry')
         
        INNER JOIN mart.dim_location loc
          ON loc.grid_cell_id = a.grid_cell_id
         AND loc.local_authority = COALESCE(a.local_authority_code, 'Unknown')
         AND loc.police_force = CONCAT('Police Force ', COALESCE(a.police_force_code, 0))
         AND loc.urban_rural = COALESCE(a.urban_rural_label, 'Urban')
         
        LEFT JOIN staging.weather_hourly w
          ON a.grid_cell_id = w.grid_cell_id
         AND DATE_TRUNC('hour', a.accident_ts) = w.weather_timestamp
         
        LEFT JOIN monthly_weather_baseline b
          ON EXTRACT(MONTH FROM a.accident_ts) = b.month_num
          
        INNER JOIN mart.dim_weather w_dim
          ON w_dim.weather_condition = COALESCE(a.weather_label, 'Fine no high winds')
         AND w_dim.light_condition = COALESCE(a.light_label, 'Daylight')
         AND w_dim.temp_category = (
             CASE 
                 WHEN COALESCE(w.temperature_2m, b.avg_temp, 11.5) < 0 THEN 'Freezing'
                 WHEN COALESCE(w.temperature_2m, b.avg_temp, 11.5) < 10 THEN 'Cold'
                 WHEN COALESCE(w.temperature_2m, b.avg_temp, 11.5) < 18 THEN 'Mild'
                 WHEN COALESCE(w.temperature_2m, b.avg_temp, 11.5) < 25 THEN 'Warm'
                 ELSE 'Hot'
             END
         )
         AND w_dim.rain_category = (
             CASE 
                 WHEN COALESCE(w.precipitation, b.avg_precip, 0.0) = 0 THEN 'None'
                 WHEN COALESCE(w.precipitation, b.avg_precip, 0.0) <= 2.5 THEN 'Light'
                 WHEN COALESCE(w.precipitation, b.avg_precip, 0.0) <= 7.5 THEN 'Moderate'
                 ELSE 'Heavy'
             END
         )
         AND w_dim.wind_category = (
             CASE 
                 WHEN COALESCE(w.wind_speed_10m, b.avg_wind, 14.2) < 10 THEN 'Calm'
                 WHEN COALESCE(w.wind_speed_10m, b.avg_wind, 14.2) <= 30 THEN 'Breeze'
                 ELSE 'Gale'
             END
         );
        """
    )
    with engine.begin() as conn:
        conn.execute(sql)
        count = conn.execute(text("SELECT COUNT(*) FROM mart.fact_accident;")).scalar()
    logger.info(f"mart.fact_accident populated with {count:,} fact records.")
    return count


def build_model_ready_table() -> int:
    """
    Populate mart.model_ready: Wide, flat, completely leak-free table.
    Enforces strict exclusion of casualty outcomes and post-collision data.
    """
    logger.info("Building mart.model_ready table (leakage-free)...")
    engine = get_db_engine()
    sql = text(
        """
        TRUNCATE TABLE mart.model_ready;
        
        WITH monthly_weather_baseline AS (
            SELECT 
                EXTRACT(MONTH FROM weather_timestamp) AS month_num,
                AVG(temperature_2m) AS avg_temp,
                AVG(precipitation) AS avg_precip,
                AVG(wind_speed_10m) AS avg_wind
            FROM staging.weather_hourly
            GROUP BY 1
        )
        INSERT INTO mart.model_ready (
            accident_id, accident_ts, hour, weekday, month, is_weekend, is_rush_hour,
            road_type, speed_limit, junction_detail, road_surface, light_condition,
            weather_condition, urban_or_rural, grid_cell_id, temperature_2m, precipitation,
            wind_speed_10m, num_vehicles, severity_code, severity_label
        )
        SELECT 
            a.accident_id,
            a.accident_ts,
            EXTRACT(HOUR FROM a.accident_ts)::smallint AS hour,
            a.day_of_week AS weekday,
            EXTRACT(MONTH FROM a.accident_ts)::smallint AS month,
            (a.day_of_week IN (1, 7)) AS is_weekend,
            (EXTRACT(HOUR FROM a.accident_ts) BETWEEN 7 AND 9 OR EXTRACT(HOUR FROM a.accident_ts) BETWEEN 16 AND 18) AS is_rush_hour,
            
            COALESCE(a.road_type_label, 'Single carriageway') AS road_type,
            COALESCE(a.speed_limit, 30)::smallint AS speed_limit,
            COALESCE(a.junction_detail_label, 'Not at junction') AS junction_detail,
            COALESCE(a.road_surface_label, 'Dry') AS road_surface,
            COALESCE(a.light_label, 'Daylight') AS light_condition,
            COALESCE(a.weather_label, 'Fine no high winds') AS weather_condition,
            COALESCE(a.urban_rural_label, 'Urban') AS urban_or_rural,
            
            a.grid_cell_id,
            COALESCE(w.temperature_2m, b.avg_temp, 11.5) AS temperature_2m,
            COALESCE(w.precipitation, b.avg_precip, 0.0) AS precipitation,
            COALESCE(w.wind_speed_10m, b.avg_wind, 14.2) AS wind_speed_10m,
            
            -- Only context known at collision time (NO casualties!)
            a.num_vehicles,
            
            -- Multiclass Target
            a.severity_code,
            a.severity_label
            
        FROM clean.accident a
        LEFT JOIN staging.weather_hourly w
          ON a.grid_cell_id = w.grid_cell_id
         AND DATE_TRUNC('hour', a.accident_ts) = w.weather_timestamp
        LEFT JOIN monthly_weather_baseline b
          ON EXTRACT(MONTH FROM a.accident_ts) = b.month_num;
        """
    )
    with engine.begin() as conn:
        conn.execute(sql)
        count = conn.execute(text("SELECT COUNT(*) FROM mart.model_ready;")).scalar()
    logger.info(f"mart.model_ready populated with {count:,} leak-free records.")
    return count


def run_build_mart() -> Dict[str, int]:
    """Execute full dimensional modeling workflow."""
    start_time = time.time()
    logger.info("=== Starting Mart Schema & Model-Ready Build Pipeline ===")

    dim_date_cnt = build_dim_date()
    dim_road_cnt = build_dim_road()
    dim_weather_cnt = build_dim_weather()
    dim_location_cnt = build_dim_location()
    fact_cnt = build_fact_accident()
    model_ready_cnt = build_model_ready_table()

    elapsed = time.time() - start_time
    summary = {
        "dim_date_count": dim_date_cnt,
        "dim_road_count": dim_road_cnt,
        "dim_weather_count": dim_weather_cnt,
        "dim_location_count": dim_location_cnt,
        "fact_accident_count": fact_cnt,
        "model_ready_count": model_ready_cnt,
        "duration_seconds": round(elapsed, 2)
    }
    logger.info(f"=== Mart Build Pipeline Complete in {elapsed:.2f}s ===")
    return summary


if __name__ == "__main__":
    summary = run_build_mart()
    print("\n--- Mart & Model-Ready Build Summary ---")
    for k, v in summary.items():
        print(f"  {k}: {v}")
