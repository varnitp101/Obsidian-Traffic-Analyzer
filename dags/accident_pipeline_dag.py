"""
DAG: accident_pipeline_dag.py
Description: End-to-end Apache Airflow orchestration DAG for UK Traffic Accident Analytics & MLOps Platform.
Task Flow:
  [extract_accidents, extract_weather]
                 │
                 ▼
           load_staging
                 │
                 ▼
          clean_validate
                 │
                 ▼
           join_weather
                 │
                 ▼
            build_mart
                 │
                 ▼
             dq_report
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to sys.path so Airflow workers resolve src modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Try importing Airflow components safely (Airflow executes in Docker/Linux containers)
try:
    from airflow import DAG  # type: ignore[import-not-found, import-untyped]
    from airflow.operators.python import PythonOperator  # type: ignore[import-not-found, import-untyped]
    AIRFLOW_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    AIRFLOW_AVAILABLE = False
    DAG = None  # type: ignore
    PythonOperator = None  # type: ignore

logger = logging.getLogger("airflow.accident_pipeline")


# -----------------------------------------------------------------------------
# Task Callables
# -----------------------------------------------------------------------------
def task_extract_accidents(**context) -> dict:
    """Task 1: Ingest raw STATS19 CSVs with exponential backoff & retry."""
    from src.ingestion.download_stats19 import download_stats19_year
    logger.info("Executing task: extract_accidents")
    year = 2022
    summary = download_stats19_year(year=year)
    logger.info(f"extract_accidents finished with summary: {summary}")
    return {"status": "success", "year": year, "files": list(summary.keys())}


def task_extract_weather(**context) -> dict:
    """Task 2: Ingest historical Open-Meteo weather by grid centroids."""
    from src.ingestion.fetch_weather import run_weather_ingestion
    logger.info("Executing task: extract_weather")
    # Ingest top 50 accident-dense grid cells for 2022
    total_records = run_weather_ingestion(
        start_date="2022-01-01",
        end_date="2022-12-31",
        batch_size=25,
        max_cells=50,
        truncate_staging=False
    )
    logger.info(f"extract_weather finished with {total_records:,} records.")
    return {"status": "success", "records_loaded": total_records}


def task_load_staging(**context) -> dict:
    """Task 3: Bulk copy raw CSVs into staging.collision, staging.vehicle, staging.casualty."""
    from src.ingestion.load_staging import load_all_staging
    logger.info("Executing task: load_staging")
    results = load_all_staging(year=2022)
    logger.info(f"load_staging finished with results: {results}")
    return {"status": "success", "tables": results}


def task_clean_validate(**context) -> dict:
    """Task 4: Evaluate DQ rules, log rejected records, decode codes, load clean schema."""
    from src.etl.clean import run_clean_and_validate
    logger.info("Executing task: clean_validate")
    summary = run_clean_and_validate(truncate_clean=True)
    logger.info(f"clean_validate finished with summary: {summary}")
    return summary


def task_join_weather(**context) -> dict:
    """Task 5: Execute spatial-temporal join, evaluate match rate, log to ops.dq_results."""
    from src.etl.join_weather import evaluate_weather_join_coverage
    import time
    logger.info("Executing task: join_weather")
    run_id = f"dag_join_{int(time.time())}"
    stats = evaluate_weather_join_coverage(run_id=run_id)
    logger.info(f"join_weather finished: Match={stats['matched_accidents']:,} ({stats['match_percentage']}%)")
    return stats


def task_build_mart(**context) -> dict:
    """Task 6: Populate Star Schema (fact + dimensions) and leak-free mart.model_ready table."""
    from src.etl.build_mart import run_build_mart
    logger.info("Executing task: build_mart")
    summary = run_build_mart()
    logger.info(f"build_mart finished with summary: {summary}")
    return summary


def task_dq_report(**context) -> dict:
    """Task 7: Evaluate global pipeline health and log pipeline completion."""
    from sqlalchemy import text
    from src.db.connection import get_db_engine
    import pandas as pd
    
    logger.info("Executing task: dq_report")
    engine = get_db_engine()
    with engine.connect() as conn:
        clean_acc_cnt = conn.execute(text("SELECT COUNT(*) FROM clean.accident;")).scalar()
        fact_acc_cnt = conn.execute(text("SELECT COUNT(*) FROM mart.fact_accident;")).scalar()
        model_ready_cnt = conn.execute(text("SELECT COUNT(*) FROM mart.model_ready;")).scalar()
        rejected_cnt = conn.execute(text("SELECT COUNT(*) FROM ops.rejected_records;")).scalar()
        
    report = {
        "status": "HEALTHY",
        "clean_accidents": clean_acc_cnt,
        "fact_accidents": fact_acc_cnt,
        "model_ready_records": model_ready_cnt,
        "total_rejected_records": rejected_cnt,
    }
    logger.info(f"=== PIPELINE HEALTH REPORT ===\n{report}")
    return report


# -----------------------------------------------------------------------------
# Airflow DAG Definition
# -----------------------------------------------------------------------------
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

if AIRFLOW_AVAILABLE:
    with DAG(
        dag_id="accident_pipeline_dag",
        default_args=default_args,
        description="End-to-end UK Traffic Accident Ingestion, Validation, Cleaning & Modeling Mart Pipeline",
        schedule_interval="@weekly",
        start_date=datetime(2023, 1, 1),
        catchup=False,
        max_active_runs=1,
        tags=["traffic", "stats19", "weather", "mlops", "data_engineering"],
    ) as dag:

        t_extract_accidents = PythonOperator(
            task_id="extract_accidents",
            python_callable=task_extract_accidents,
        )

        t_extract_weather = PythonOperator(
            task_id="extract_weather",
            python_callable=task_extract_weather,
        )

        t_load_staging = PythonOperator(
            task_id="load_staging",
            python_callable=task_load_staging,
        )

        t_clean_validate = PythonOperator(
            task_id="clean_validate",
            python_callable=task_clean_validate,
        )

        t_join_weather = PythonOperator(
            task_id="join_weather",
            python_callable=task_join_weather,
        )

        t_build_mart = PythonOperator(
            task_id="build_mart",
            python_callable=task_build_mart,
        )

        t_dq_report = PythonOperator(
            task_id="dq_report",
            python_callable=task_dq_report,
        )

        # Dependency Graph
        [t_extract_accidents, t_extract_weather] >> t_load_staging >> t_clean_validate >> t_join_weather >> t_build_mart >> t_dq_report

else:
    dag = None


# -----------------------------------------------------------------------------
# Standalone Local Verification Runner
# -----------------------------------------------------------------------------
def run_pipeline_locally():
    """Execute all tasks sequentially for local verification outside Airflow."""
    print("=" * 70)
    print("RUNNING ACCIDENT PIPELINE END-TO-END (LOCAL VERIFICATION)")
    print("=" * 70)
    
    print("\n[1/7] Running task: extract_accidents...")
    res_acc = task_extract_accidents()
    print("  ->", res_acc)
    
    print("\n[2/7] Running task: extract_weather...")
    res_w = task_extract_weather()
    print("  ->", res_w)
    
    print("\n[3/7] Running task: load_staging...")
    res_stage = task_load_staging()
    print("  ->", res_stage)
    
    print("\n[4/7] Running task: clean_validate...")
    res_clean = task_clean_validate()
    print("  ->", res_clean)
    
    print("\n[5/7] Running task: join_weather...")
    res_join = task_join_weather()
    print("  ->", res_join)
    
    print("\n[6/7] Running task: build_mart...")
    res_mart = task_build_mart()
    print("  ->", res_mart)
    
    print("\n[7/7] Running task: dq_report...")
    res_dq = task_dq_report()
    print("  ->", res_dq)
    
    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION COMPLETE - ALL TASKS PASSED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline_locally()
