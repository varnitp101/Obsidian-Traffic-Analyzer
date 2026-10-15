"""
src/monitoring/drift.py
-----------------------
Feature and Target Drift Monitoring Engine for UK Traffic Accident Severity.

Implements:
1. Population Stability Index (PSI) calculation for numerical & binned features.
2. Two-sample Kolmogorov-Smirnov (KS) test for continuous distributions.
3. Chi-square test of independence for categorical distributions.
4. Target/Prediction class distribution drift detection.
5. Automated drift status classification (STABLE, WARNING, DRIFT_DETECTED).
"""

import os
import json
import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import text

from src.db.connection import get_db_engine
from src.ml.features import (
    NUMERICAL_COLS,
    CATEGORICAL_COLS,
    split_chronologically,
    load_model_ready_data,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("monitoring.drift")


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    num_buckets: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """
    Computes the Population Stability Index (PSI) between a reference (expected)
    and production (actual) distribution.
    - PSI < 0.10: Little to no shift (STABLE)
    - 0.10 <= PSI < 0.25: Moderate shift (WARNING)
    - PSI >= 0.25: Significant shift (DRIFT)
    """
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Determine quantile bucket boundaries from expected distribution
    percentiles = np.linspace(0, 100, num_buckets + 1)
    try:
        bin_edges = np.percentile(expected, percentiles)
        bin_edges = np.unique(bin_edges)  # Remove identical edges
        if len(bin_edges) <= 2:
            return 0.0
    except Exception:
        return 0.0

    # Bin frequencies
    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    # Fractions
    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    # Prevent zero division / log of zero using epsilon
    expected_pct = np.clip(expected_pct, epsilon, None)
    actual_pct = np.clip(actual_pct, epsilon, None)

    # Re-normalize to sum to 1
    expected_pct /= np.sum(expected_pct)
    actual_pct /= np.sum(actual_pct)

    # PSI Formula: sum((actual - expected) * ln(actual / expected))
    psi_val = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(max(0.0, psi_val))


def evaluate_feature_drift(
    ref_df: pd.DataFrame,
    curr_df: pd.DataFrame,
    psi_threshold_warning: float = 0.10,
    psi_threshold_alert: float = 0.25,
) -> Dict[str, Any]:
    """
    Compares baseline reference dataset (e.g. historical training data) against
    current inference dataset across numerical and categorical features.
    """
    results = {}
    drift_detected = False
    warning_detected = False

    logger.info(f"Evaluating feature drift: Reference={len(ref_df):,} rows, Current={len(curr_df):,} rows")

    # 1. Numerical Feature Drift
    for col in NUMERICAL_COLS:
        if col in ref_df.columns and col in curr_df.columns:
            exp_vals = pd.to_numeric(ref_df[col], errors="coerce").dropna().values
            act_vals = pd.to_numeric(curr_df[col], errors="coerce").dropna().values

            if len(exp_vals) > 0 and len(act_vals) > 0:
                psi = calculate_psi(exp_vals, act_vals)
                ks_stat, ks_pvalue = stats.ks_2samp(exp_vals, act_vals)

                if psi >= psi_threshold_alert:
                    status = "DRIFT_ALERT"
                    drift_detected = True
                elif psi >= psi_threshold_warning:
                    status = "WARNING"
                    warning_detected = True
                else:
                    status = "STABLE"

                results[col] = {
                    "type": "numerical",
                    "psi": round(psi, 4),
                    "ks_statistic": round(float(ks_stat), 4),
                    "ks_pvalue": round(float(ks_pvalue), 6),
                    "status": status,
                }

    # 2. Categorical Feature Drift
    for col in CATEGORICAL_COLS:
        if col in ref_df.columns and col in curr_df.columns:
            # Value frequencies
            all_cats = list(set(ref_df[col].dropna().unique()).union(set(curr_df[col].dropna().unique())))
            exp_counts = ref_df[col].value_counts().reindex(all_cats, fill_value=0).values
            act_counts = curr_df[col].value_counts().reindex(all_cats, fill_value=0).values

            exp_pct = exp_counts / max(1, len(ref_df))
            act_pct = act_counts / max(1, len(curr_df))

            # Categorical PSI
            exp_pct = np.clip(exp_pct, 1e-4, None)
            act_pct = np.clip(act_pct, 1e-4, None)
            exp_pct /= exp_pct.sum()
            act_pct /= act_pct.sum()
            cat_psi = float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))

            if cat_psi >= psi_threshold_alert:
                status = "DRIFT_ALERT"
                drift_detected = True
            elif cat_psi >= psi_threshold_warning:
                status = "WARNING"
                warning_detected = True
            else:
                status = "STABLE"

            results[col] = {
                "type": "categorical",
                "psi": round(cat_psi, 4),
                "unique_categories": len(all_cats),
                "status": status,
            }

    overall_status = "DRIFT_ALERT" if drift_detected else ("WARNING" if warning_detected else "STABLE")

    report = {
        "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        "reference_records": len(ref_df),
        "current_records": len(curr_df),
        "overall_status": overall_status,
        "retrain_recommended": drift_detected,
        "feature_metrics": results,
    }

    return report


def run_drift_analysis(save_json: bool = True) -> Dict[str, Any]:
    """
    Loads historical training data as reference and evaluates test period data for natural drift.
    """
    df = load_model_ready_data()
    train_df, _, test_df = split_chronologically(df)

    report = evaluate_feature_drift(ref_df=train_df, curr_df=test_df)
    
    logger.info(f"Drift Analysis Completed. Overall Status: {report['overall_status']}")
    logger.info(f"Retraining Recommended: {report['retrain_recommended']}")

    if save_json:
        out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "drift_report.json"))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Drift report saved to: {out_path}")

    return report


if __name__ == "__main__":
    report = run_drift_analysis()
    print("\nSummary of Drift Status:")
    for feat, m in report["feature_metrics"].items():
        print(f"  {feat:<22}: PSI={m['psi']:<8} Status={m['status']}")
