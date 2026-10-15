"""
src/ml/train.py
---------------
Model training and MLflow tracking pipeline for UK Traffic Accident Severity Prediction.

Implements:
1. Chronological Train/Val/Test splitting (zero temporal leakage).
2. Class-imbalance-aware training (Class weighting for Fatal=1, Serious=2, Slight=3).
3. Multiple candidate architectures:
   - Baseline Stratified Dummy
   - Balanced Logistic Regression
   - Balanced Random Forest
   - Class-Weighted LightGBM
   - XGBoost Classifier
4. MLflow Experiment Tracking: logs params, metrics, confusion matrix plots, and models.
5. Champion Model Selection and export to `models/champion_pipeline.joblib`.
"""

import os
import time
import logging
from typing import Dict, Any, List
import joblib
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn

from src.ml.features import (
    load_model_ready_data,
    split_chronologically,
    extract_features_and_target,
    build_preprocessor,
    SEVERITY_NAMES,
)
from src.ml.evaluate import compute_metrics, plot_confusion_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("ml.train")

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))
os.makedirs(MODELS_DIR, exist_ok=True)
MLFLOW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "mlruns"))
os.makedirs(MLFLOW_DIR, exist_ok=True)


def configure_mlflow(experiment_name: str = "TrafficAccident_Severity_Prediction"):
    """
    Configures MLflow tracking URI using SQLite database backend and initializes experiment.
    """
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "mlflow.db"))
    tracking_uri = f"sqlite:///{db_path.replace(os.sep, '/')}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow configured with DB Backend URI: {tracking_uri}, Experiment: {experiment_name}")


def get_model_candidates(class_weights: Dict[int, float]) -> Dict[str, Any]:
    """
    Returns candidate model instances configured for imbalanced multi-class prediction.
    """
    candidates = {
        "Baseline_Stratified": DummyClassifier(strategy="stratified", random_state=42),
        "Logistic_Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=500,
            random_state=42,
        ),
        "Random_Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=120,
            learning_rate=0.05,
            max_depth=7,
            num_leaves=31,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.08,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        ),
    }
    return candidates


def train_and_evaluate_all():
    """
    Main training workflow executing chronological training, candidate evaluation,
    MLflow tracking, and champion model persistence.
    """
    start_time = time.time()
    configure_mlflow()

    # 1. Load data & chronological split
    df = load_model_ready_data()
    train_df, val_df, test_df = split_chronologically(df)

    X_train, y_train = extract_features_and_target(train_df)
    X_val, y_val = extract_features_and_target(val_df)
    X_test, y_test = extract_features_and_target(test_df)

    # Compute class weights for reference
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, weights))
    logger.info(f"Computed Balanced Class Weights: { {SEVERITY_NAMES[k]: round(v, 3) for k, v in class_weight_dict.items()} }")

    candidates = get_model_candidates(class_weight_dict)
    results = []

    champion_name = None
    champion_val_macro_f1 = -1.0
    champion_pipeline = None

    for model_name, classifier in candidates.items():
        logger.info(f"\n{'='*25} Training: {model_name} {'='*25}")
        
        # Build unified Pipeline to eliminate train-serve skew
        preprocessor = build_preprocessor()
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ])

        with mlflow.start_run(run_name=model_name):
            t0 = time.time()
            # Fit pipeline
            pipeline.fit(X_train, y_train)
            train_duration = time.time() - t0

            # Predict on Validation
            val_preds = pipeline.predict(X_val)
            val_probs = pipeline.predict_proba(X_val) if hasattr(pipeline, "predict_proba") else None
            val_metrics = compute_metrics(y_val, val_preds, val_probs, prefix="val")

            # Predict on Test
            test_preds = pipeline.predict(X_test)
            test_probs = pipeline.predict_proba(X_test) if hasattr(pipeline, "predict_proba") else None
            test_metrics = compute_metrics(y_test, test_preds, test_probs, prefix="test")

            # Log parameters
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("val_samples", len(X_val))
            mlflow.log_param("test_samples", len(X_test))
            mlflow.log_param("train_duration_sec", round(train_duration, 2))
            
            # Log metrics
            all_metrics = {**val_metrics, **test_metrics, "train_duration_sec": train_duration}
            for k, v in all_metrics.items():
                mlflow.log_metric(k, v)

            # Confusion Matrix Plot Artifact
            cm_path = os.path.join(MODELS_DIR, f"cm_{model_name.lower()}.png")
            plot_confusion_matrix(y_test, test_preds, cm_path, title=f"Confusion Matrix - {model_name}")
            mlflow.log_artifact(cm_path, artifact_path="evaluation_plots")

            # Log Scikit-learn Model with cloudpickle
            mlflow.sklearn.log_model(pipeline, name="model", serialization_format="cloudpickle")

            logger.info(f"[{model_name}] Val Macro-F1: {val_metrics['val_macro_f1']:.4f} | "
                        f"Test Macro-F1: {test_metrics['test_macro_f1']:.4f} | "
                        f"Fatal Recall: {test_metrics['test_fatal_recall']:.2%} | "
                        f"Serious Recall: {test_metrics['test_serious_recall']:.2%}")

            results.append({
                "model": model_name,
                "val_macro_f1": val_metrics["val_macro_f1"],
                "test_macro_f1": test_metrics["test_macro_f1"],
                "fatal_recall": test_metrics["test_fatal_recall"],
                "serious_recall": test_metrics["test_serious_recall"],
                "slight_recall": test_metrics["test_slight_recall"],
                "duration_s": round(train_duration, 1),
            })

            # Track champion based on Validation Macro-F1
            if val_metrics["val_macro_f1"] > champion_val_macro_f1:
                champion_val_macro_f1 = val_metrics["val_macro_f1"]
                champion_name = model_name
                champion_pipeline = pipeline

    # Summary table
    summary_df = pd.DataFrame(results).sort_values(by="val_macro_f1", ascending=False)
    logger.info(f"\n\n{'='*30} MODEL COMPARISON SUMMARY {'='*30}\n{summary_df.to_string(index=False)}")

    # Save Champion Model
    champion_path = os.path.join(MODELS_DIR, "champion_pipeline.joblib")
    joblib.dump(champion_pipeline, champion_path)
    logger.info(f"\n[CHAMPION MODEL]: {champion_name} (Val Macro-F1: {champion_val_macro_f1:.4f})")
    logger.info(f"Persisted Champion Pipeline to: {champion_path}")

    # Register in MLflow Model Registry if available
    try:
        model_uri = f"runs:/{mlflow.last_active_run().info.run_id}/model"
        mlflow.register_model(model_uri, "TrafficAccidentSeverityModel")
        logger.info("Successfully registered champion model in MLflow Model Registry.")
    except Exception as e:
        logger.info(f"Local MLflow registry note: {e}")

    total_time = time.time() - start_time
    logger.info(f"Complete Training and MLflow Tracking finished in {total_time:.2f}s.")
    return summary_df


if __name__ == "__main__":
    train_and_evaluate_all()
