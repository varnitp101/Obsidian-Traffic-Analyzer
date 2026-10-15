"""
Module: fetch_weather.py
Description: Resilient, grid-clustered historical weather ingestion from Open-Meteo API.
Groups collisions into 0.25° spatial grid centroids (~25 km resolution), fetches hourly
meteorological data for the full year range in batch HTTP calls with exponential backoff,
saves raw API responses untouched in data/raw/weather/, and bulk-loads into staging.weather_hourly.
"""

import os
import sys
import json
import time
import uuid
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
from sqlalchemy import text

from src.db.connection import get_db_engine
from src.db.load import bulk_copy_df, log_ingestion_event

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ingestion.weather")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_WEATHER_DIR = REPO_ROOT / "data" / "raw" / "weather"
OPEN_METEO_ARCHIVE_URL = os.getenv("OPEN_METEO_BASE_URL", "https://archive-api.open-meteo.com/v1/archive")
GRID_SIZE = float(os.getenv("GRID_CELL_SIZE_DEGREES", "0.25"))


def get_retry_session(
    retries: int = 5,
    backoff_factor: float = 2.0,
    status_forcelist: tuple = (429, 500, 502, 503, 504)
) -> requests.Session:
    """Create a requests session with exponential backoff and rate-limit handling."""
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def compute_grid_centroids(
    uk_bounds: Tuple[float, float, float, float] = (49.0, 61.0, -9.0, 2.0)
) -> pd.DataFrame:
    """
    Extract distinct 0.25° spatial grid centroids from staging.collision.
    Only includes valid coordinates within the UK bounding box.
    """
    engine = get_db_engine()
    min_lat, max_lat, min_lon, max_lon = uk_bounds

    query = text(
        r"""
        SELECT 
            ROUND(latitude::numeric / :grid_size) * :grid_size AS grid_lat,
            ROUND(longitude::numeric / :grid_size) * :grid_size AS grid_lon,
            COUNT(*) AS accident_count
        FROM staging.collision
        WHERE latitude IS NOT NULL 
          AND longitude IS NOT NULL
          AND latitude ~ '^-?[0-9]+(\.[0-9]+)?$'
          AND longitude ~ '^-?[0-9]+(\.[0-9]+)?$'
          AND latitude::numeric BETWEEN :min_lat AND :max_lat
          AND longitude::numeric BETWEEN :min_lon AND :max_lon
          AND NOT (latitude::numeric = 0 AND longitude::numeric = 0)
        GROUP BY 1, 2
        ORDER BY accident_count DESC;
        """
    )

    with engine.connect() as conn:
        df = pd.read_sql(
            query,
            conn,
            params={
                "grid_size": GRID_SIZE,
                "min_lat": min_lat,
                "max_lat": max_lat,
                "min_lon": min_lon,
                "max_lon": max_lon
            }
        )

    df["grid_lat"] = df["grid_lat"].astype(float)
    df["grid_lon"] = df["grid_lon"].astype(float)
    df["grid_cell_id"] = df.apply(
        lambda r: f"GRID_{r['grid_lat']:.2f}_{r['grid_lon']:.2f}", axis=1
    )
    df = df.drop_duplicates(subset=["grid_cell_id"]).sort_values(by=["accident_count"], ascending=False).reset_index(drop=True)

    logger.info(f"Identified {len(df):,} unique {GRID_SIZE}° grid cells across valid collision records.")
    return df


def fetch_weather_batch(
    session: requests.Session,
    centroids_batch: pd.DataFrame,
    start_date: str = "2022-01-01",
    end_date: str = "2022-12-31",
    raw_save_path: Optional[Path] = None,
) -> List[Dict]:
    """
    Fetch hourly historical weather from Open-Meteo for a batch of coordinates.
    Saves the pristine JSON response to data/raw/weather/ if path provided.
    """
    lats = centroids_batch["grid_lat"].tolist()
    lons = centroids_batch["grid_lon"].tolist()

    params = {
        "latitude": lats,
        "longitude": lons,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,precipitation,wind_speed_10m,weather_code,cloud_cover",
        "timezone": "UTC"
    }

    response = session.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=60)
    response.raise_for_status()
    raw_data = response.json()

    # If single coordinate requested, Open-Meteo returns a dict; normalize to list
    if isinstance(raw_data, dict):
        raw_data = [raw_data]

    # Save untouched raw JSON response
    if raw_save_path:
        raw_save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(raw_save_path, "w", encoding="utf-8") as f:
            json.dump(raw_data, f, indent=2)

    return raw_data


def parse_weather_responses(
    responses: List[Dict],
    centroids_batch: pd.DataFrame
) -> pd.DataFrame:
    """
    Parse Open-Meteo JSON responses into a flat DataFrame matching staging.weather_hourly schema.
    Responses map 1-to-1 by order index to the requested centroids_batch rows.
    """
    records = []
    centroids_list = centroids_batch.to_dict(orient="records")

    for i, item in enumerate(responses):
        if i >= len(centroids_list):
            break
        centroid = centroids_list[i]
        grid_cell_id = centroid["grid_cell_id"]
        lat = centroid["grid_lat"]
        lon = centroid["grid_lon"]

        hourly = item.get("hourly", {})
        times = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])
        precips = hourly.get("precipitation", [])
        winds = hourly.get("wind_speed_10m", [])
        w_codes = hourly.get("weather_code", [])
        clouds = hourly.get("cloud_cover", [])

        n = len(times)
        for t_idx in range(n):
            records.append({
                "grid_cell_id": grid_cell_id,
                "grid_latitude": lat,
                "grid_longitude": lon,
                "weather_timestamp": times[t_idx].replace("T", " "),
                "temperature_2m": temps[t_idx] if t_idx < len(temps) else None,
                "precipitation": precips[t_idx] if t_idx < len(precips) else None,
                "wind_speed_10m": winds[t_idx] if t_idx < len(winds) else None,
                "weather_code": w_codes[t_idx] if t_idx < len(w_codes) else None,
                "cloud_cover": clouds[t_idx] if t_idx < len(clouds) else None,
            })

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.drop_duplicates(subset=["grid_cell_id", "weather_timestamp"])
    return df



def run_weather_ingestion(
    start_date: str = "2022-01-01",
    end_date: str = "2022-12-31",
    batch_size: int = 25,
    max_cells: Optional[int] = None,
    truncate_staging: bool = False
) -> int:
    """
    Orchestrates end-to-end weather ingestion across UK grid cells.
    """
    run_id = f"weather_{start_date[:4]}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    session = get_retry_session()
    RAW_WEATHER_DIR.mkdir(parents=True, exist_ok=True)

    centroids = compute_grid_centroids()
    if max_cells:
        centroids = centroids.head(max_cells)
        logger.info(f"Limited run to {max_cells} grid cells for development / verification.")

    # Check if staging.weather_hourly already contains sufficient records
    engine = get_db_engine()
    with engine.connect() as conn:
        existing = conn.execute(text("SELECT COUNT(*) FROM staging.weather_hourly;")).scalar() or 0
    if existing >= (len(centroids) * 8000) and not truncate_staging:
        logger.info(f"staging.weather_hourly already populated with {existing:,} records. Skipping re-fetch.")
        return existing

    total_cells = len(centroids)
    total_rows_loaded = 0
    num_batches = (total_cells + batch_size - 1) // batch_size

    logger.info(f"Starting weather ingestion for {total_cells} cells ({num_batches} API batches) from {start_date} to {end_date}...")

    first_batch = True
    for b_idx in range(num_batches):
        start_idx = b_idx * batch_size
        end_idx = min(start_idx + batch_size, total_cells)
        batch_df = centroids.iloc[start_idx:end_idx].copy()

        batch_raw_file = RAW_WEATHER_DIR / f"weather_raw_batch_{b_idx+1:04d}.json"
        logger.info(f"Batch {b_idx+1}/{num_batches}: fetching {len(batch_df)} cells...")

        try:
            raw_responses = fetch_weather_batch(
                session=session,
                centroids_batch=batch_df,
                start_date=start_date,
                end_date=end_date,
                raw_save_path=batch_raw_file,
            )

            parsed_df = parse_weather_responses(raw_responses, batch_df)
            if not parsed_df.empty:
                loaded = bulk_copy_df(
                    df=parsed_df,
                    schema_name="staging",
                    table_name="weather_hourly",
                    truncate_first=(truncate_staging and first_batch),
                    run_id=f"{run_id}_b{b_idx+1}",
                    source_name=f"open_meteo_weather_{start_date[:4]}"
                )
                total_rows_loaded += loaded
                first_batch = False

            # Gentle pacing to respect Open-Meteo free-tier guidelines
            time.sleep(1.0)

        except Exception as e:
            err_msg = f"Failed on batch {b_idx+1}: {e}"
            logger.error(err_msg, exc_info=True)
            log_ingestion_event(
                run_id=f"{run_id}_err",
                source_name=f"open_meteo_weather_{start_date[:4]}",
                status="failed",
                records_extracted=total_rows_loaded,
                error_message=err_msg
            )
            raise

    log_ingestion_event(
        run_id=run_id,
        source_name=f"open_meteo_weather_{start_date[:4]}",
        status="success",
        records_extracted=total_rows_loaded,
    )

    logger.info(f"Completed weather ingestion! Total hourly weather records loaded: {total_rows_loaded:,}")
    return total_rows_loaded


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch historical weather by spatial grid centroids.")
    parser.add_argument("--start", default="2022-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2022-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--batch-size", type=int, default=25, help="Coordinates per API call")
    parser.add_argument("--max-cells", type=int, default=None, help="Cap number of cells for testing")
    parser.add_argument("--truncate", action="store_true", help="Truncate staging.weather_hourly first")

    args = parser.parse_args()
    run_weather_ingestion(
        start_date=args.start,
        end_date=args.end,
        batch_size=args.batch_size,
        max_cells=args.max_cells,
        truncate_staging=args.truncate
    )
