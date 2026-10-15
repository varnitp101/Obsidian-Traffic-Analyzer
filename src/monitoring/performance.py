"""
src/monitoring/performance.py
-----------------------------
Model performance, latency monitoring, and automated retraining policy evaluator
for the UK Traffic Accident Severity MLOps platform.

Features:
1. Operational Telemetry: Queries ops.prediction_log to track real-time p50/p95/p99 latency.
2. Prediction Distribution Monitor: Alerts on class collapse or unnatural skew.
3. Automated Retraining Decision Engine: Evaluates drift reports + performance thresholds to trigger retraining.
"""

import os
import json
import logging
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sqlalchemy import text

from src.db.connection import get_db_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("monitoring.performance")


def get_prediction_latency_stats() -> Dict[str, Any]:
    """
    Computes latency percentiles and volume metrics from ops.prediction_log.
    """
    engine = get_db_engine()
    query = """
    SELECT latency_ms, predicted_severity_code, timestamp
    FROM ops.prediction_log
    ORDER BY timestamp DESC
    LIMIT 10000;
    """
    try:
        df = pd.read_sql(query, engine)
        if df.empty:
            return {
                "total_inferences": 0,
                "latency_p50_ms": 0.0,
                "latency_p95_ms": 0.0,
                "latency_p99_ms": 0.0,
                "class_distribution": {"Fatal": 0, "Serious": 0, "Slight": 0},
            }

        latencies = df["latency_ms"].dropna().values
        counts = df["predicted_severity_code"].value_counts().to_dict()

        return {
            "total_inferences": len(df),
            "latency_p50_ms": round(float(np.percentile(latencies, 50)), 2),
            "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2),
            "latency_p99_ms": round(float(np.percentile(latencies, 99)), 2),
            "class_distribution": {
                "Fatal": int(counts.get(1, 0)),
                "Serious": int(counts.get(2, 0)),
                "Slight": int(counts.get(3, 0)),
            },
        }
    except Exception as e:
        logger.warning(f"Could not read latency stats: {e}")
        return {"total_inferences": 0, "error": str(e)}


def evaluate_retraining_trigger(
    drift_report_path: str = None,
    max_acceptable_psi: float = 0.25,
    min_fatal_pct: float = 0.005,  # Fatal shouldn't disappear entirely (<0.5%)
    max_fatal_pct: float = 0.08,   # Fatal shouldn't explode (>8%)
) -> Dict[str, Any]:
    """
    Evaluates whether an automated retraining pipeline should be triggered.
    Triggers when:
    1. Critical feature PSI exceeds max_acceptable_psi (0.25).
    2. Predicted severity distribution displays anomalous collapse (class collapse).
    """
    if drift_report_path is None:
        drift_report_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "models", "drift_report.json")
        )

    retrain_reasons = []
    drift_status = "UNKNOWN"

    # 1. Check Drift Report
    if os.path.exists(drift_report_path):
        try:
            with open(drift_report_path, "r", encoding="utf-8") as f:
                drift_data = json.load(f)
            drift_status = drift_data.get("overall_status", "UNKNOWN")
            if drift_data.get("retrain_recommended", False):
                retrain_reasons.append(f"Feature drift detected: Status={drift_status}")
            
            # Check individual critical features
            for feat, m in drift_data.get("feature_metrics", {}).items():
                if m.get("psi", 0) >= max_acceptable_psi:
                    retrain_reasons.append(f"Severe PSI drift on '{feat}' (PSI={m.get('psi')})")
        except Exception as e:
            logger.warning(f"Could not parse drift report: {e}")

    # 2. Check Live Prediction Distribution
    latency_stats = get_prediction_latency_stats()
    total_preds = latency_stats.get("total_inferences", 0)
    if total_preds >= 50:
        dist = latency_stats.get("class_distribution", {})
        fatal_ratio = dist.get("Fatal", 0) / total_preds
        if fatal_ratio < min_fatal_pct:
            retrain_reasons.append(f"Fatal prediction ratio collapsed to {fatal_ratio:.2%} (<{min_fatal_pct:.2%})")
        elif fatal_ratio > max_fatal_pct:
            retrain_reasons.append(f"Fatal prediction ratio spiked to {fatal_ratio:.2%} (>{max_fatal_pct:.2%})")

    should_retrain = len(retrain_reasons) > 0

    decision = {
        "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        "should_retrain": should_retrain,
        "trigger_action": "TRIGGER_AIRFLOW_RETRAIN" if should_retrain else "NO_ACTION",
        "drift_status": drift_status,
        "reasons": retrain_reasons if should_retrain else ["All monitoring metrics within operational thresholds."],
        "latency_telemetry": latency_stats,
    }

    logger.info(f"Retraining Evaluation: Should Retrain = {should_retrain}")
    return decision


if __name__ == "__main__":
    dec = evaluate_retraining_trigger()
    print("\nRetraining Trigger Decision:")
    print(json.dumps(dec, indent=2))
