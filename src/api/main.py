"""
src/api/main.py
---------------
Production-grade FastAPI inference microservice for UK Traffic Accident Severity Prediction.

Features:
1. Unified scikit-learn Pipeline loading (`models/champion_pipeline.joblib`) preventing train-serve skew.
2. Modern Pydantic V2 input validation (`AccidentFeatureInput` and `BatchAccidentInput`).
3. Automated operational logging into PostgreSQL `ops.prediction_log` with latency tracking.
4. Endpoints:
   - `GET /health`: Service health, model load status, database connectivity.
   - `GET /model-info`: Model architecture, features, and target class semantics.
   - `POST /predict`: Single collision severity prediction with probabilities.
   - `POST /predict/batch`: High-throughput batch prediction.
   - `GET /metrics`: Operational metrics (prediction count, average latency, class distribution).
"""

import os
import time
import json
import uuid
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from contextlib import asynccontextmanager
import joblib
import pandas as pd
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from src.db.connection import get_db_engine
from src.ml.features import REVERSE_TARGET_MAP, SEVERITY_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("api.main")

MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "champion_pipeline.joblib"))
model_pipeline = None
start_time = datetime.now(timezone.utc)


def load_model():
    """Loads the champion scikit-learn pipeline from disk."""
    global model_pipeline
    if os.path.exists(MODEL_PATH):
        try:
            model_pipeline = joblib.load(MODEL_PATH)
            logger.info(f"Loaded champion pipeline from: {MODEL_PATH}")
        except Exception as e:
            logger.error(f"Failed to load model pipeline from {MODEL_PATH}: {e}")
            model_pipeline = None
    else:
        logger.warning(f"Champion model not found at {MODEL_PATH}. Retrain or wait for training to finish.")


# Load model immediately on module import
load_model()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern FastAPI lifespan context manager."""
    load_model()
    yield


app = FastAPI(
    title="UK Traffic Accident Severity Prediction API",
    description="Production MLOps inference microservice predicting collision severity using road, weather, and vehicular dynamics.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AccidentFeatureInput(BaseModel):
    speed_limit: float = Field(..., ge=10, le=100, json_schema_extra={"example": 30.0}, description="Speed limit in mph")
    num_vehicles: int = Field(..., ge=1, le=50, json_schema_extra={"example": 2}, description="Number of vehicles involved")
    temperature_2m: float = Field(..., ge=-30, le=50, json_schema_extra={"example": 12.5}, description="Temperature in Celsius")
    precipitation: float = Field(..., ge=0, le=200, json_schema_extra={"example": 0.2}, description="Precipitation in mm")
    wind_speed_10m: float = Field(..., ge=0, le=200, json_schema_extra={"example": 15.0}, description="Wind speed in km/h")
    
    road_type: str = Field(..., json_schema_extra={"example": "Single carriageway"}, description="Type of road")
    junction_detail: str = Field(..., json_schema_extra={"example": "T or staggered junction"}, description="Junction geometry")
    road_surface: str = Field(..., json_schema_extra={"example": "Wet or damp"}, description="Road surface condition")
    light_condition: str = Field(..., json_schema_extra={"example": "Daylight"}, description="Lighting conditions")
    weather_condition: str = Field(..., json_schema_extra={"example": "Raining no high winds"}, description="Atmospheric weather")
    urban_or_rural: str = Field(..., json_schema_extra={"example": "Urban"}, description="Urban or rural classification")
    
    hour: int = Field(..., ge=0, le=23, json_schema_extra={"example": 17}, description="Hour of accident (0-23)")
    weekday: int = Field(..., ge=1, le=7, json_schema_extra={"example": 5}, description="Day of week (1=Sun, 7=Sat or 1=Mon)")
    month: int = Field(..., ge=1, le=12, json_schema_extra={"example": 10}, description="Month (1-12)")
    is_weekend: int = Field(..., ge=0, le=1, json_schema_extra={"example": 0}, description="1 if Saturday/Sunday else 0")
    is_rush_hour: int = Field(..., ge=0, le=1, json_schema_extra={"example": 1}, description="1 if 07:00-09:00 or 16:30-19:00 else 0")


class BatchAccidentInput(BaseModel):
    accidents: List[AccidentFeatureInput]


class PredictionOutput(BaseModel):
    prediction_id: str
    severity_code: int
    severity_label: str
    probabilities: Dict[str, float]
    latency_ms: float


def log_prediction_to_db(
    model_version: str,
    latency_ms: float,
    input_features: Dict[str, Any],
    predicted_code: int,
    predicted_label: str,
    probabilities: Dict[str, float],
):
    """Logs prediction request to ops.prediction_log."""
    try:
        engine = get_db_engine()
        insert_sql = text("""
            INSERT INTO ops.prediction_log (
                timestamp, model_version, inputs, predicted_severity_code,
                predicted_severity_label, class_probabilities, latency_ms
            ) VALUES (
                NOW(), :model_version, :inputs, :predicted_severity_code,
                :predicted_severity_label, :class_probabilities, :latency_ms
            );
        """)
        with engine.begin() as conn:
            conn.execute(
                insert_sql,
                {
                    "model_version": model_version,
                    "inputs": json.dumps(input_features),
                    "predicted_severity_code": predicted_code,
                    "predicted_severity_label": predicted_label,
                    "class_probabilities": json.dumps(probabilities),
                    "latency_ms": latency_ms,
                },
            )
    except Exception as e:
        logger.warning(f"Telemetry logging error (non-fatal): {e}")


@app.get("/health", tags=["Operational"])
def health():
    """Checks service health, model status, and database connectivity."""
    db_healthy = False
    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1;"))
        db_healthy = True
    except Exception:
        pass

    return {
        "status": "healthy" if (model_pipeline is not None and db_healthy) else "degraded",
        "model_loaded": model_pipeline is not None,
        "database_connected": db_healthy,
        "uptime_seconds": (datetime.now(timezone.utc) - start_time).total_seconds(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/model-info", tags=["Operational"])
def model_info():
    """Returns architecture details, input schema, and target semantics."""
    if model_pipeline is None:
        load_model()
    
    classifier_name = "Unknown"
    if model_pipeline is not None and hasattr(model_pipeline, "named_steps"):
        classifier_name = model_pipeline.named_steps.get("classifier", type(model_pipeline)).__class__.__name__

    return {
        "model_name": "UK_TrafficAccident_Severity_Classifier",
        "algorithm": classifier_name,
        "target_mapping": {
            1: "Fatal (Injuries proving fatal within 30 days)",
            2: "Serious (Detained in hospital, fractures, severe lacerations)",
            3: "Slight (Minor cuts, sprains, bruises)",
        },
        "zero_leakage_guarantee": "Excludes all casualty injury metrics and post-collision events.",
        "input_features": list(AccidentFeatureInput.model_fields.keys()),
    }


@app.post("/predict", response_model=PredictionOutput, tags=["Inference"])
def predict(input_data: AccidentFeatureInput):
    """
    Predicts collision severity (Fatal, Serious, Slight) based on real-time accident features.
    """
    global model_pipeline
    if model_pipeline is None:
        load_model()
        if model_pipeline is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is currently training or not available on disk.",
            )

    t0 = time.time()
    feature_dict = input_data.model_dump()
    df_input = pd.DataFrame([feature_dict])

    try:
        # Predict using unified pipeline
        pred_idx = int(model_pipeline.predict(df_input)[0])
        severity_code = REVERSE_TARGET_MAP.get(pred_idx, pred_idx + 1)
        severity_label = SEVERITY_NAMES.get(pred_idx, f"Severity {severity_code}")

        # Compute probabilities
        if hasattr(model_pipeline, "predict_proba"):
            probs = model_pipeline.predict_proba(df_input)[0]
            prob_dict = {
                "Fatal": round(float(probs[0]), 4),
                "Serious": round(float(probs[1]), 4),
                "Slight": round(float(probs[2]), 4),
            }
        else:
            prob_dict = {severity_label: 1.0}

        latency_ms = round((time.time() - t0) * 1000, 2)
        pred_id = f"pred_{uuid.uuid4().hex[:12]}"

        # Log prediction to DB telemetry asynchronously
        log_prediction_to_db("v1.0", latency_ms, feature_dict, severity_code, severity_label, prob_dict)

        return PredictionOutput(
            prediction_id=pred_id,
            severity_code=severity_code,
            severity_label=severity_label,
            probabilities=prob_dict,
            latency_ms=latency_ms,
        )
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", tags=["Inference"])
def predict_batch(batch_input: BatchAccidentInput):
    """
    High-throughput batch prediction for multiple collision records.
    """
    global model_pipeline
    if model_pipeline is None:
        load_model()
        if model_pipeline is None:
            raise HTTPException(status_code=503, detail="Model pipeline unavailable.")

    t0 = time.time()
    records = [acc.model_dump() for acc in batch_input.accidents]
    df_batch = pd.DataFrame(records)

    try:
        preds = model_pipeline.predict(df_batch)
        probs = model_pipeline.predict_proba(df_batch) if hasattr(model_pipeline, "predict_proba") else None
        
        batch_results = []
        for i, pred_idx in enumerate(preds):
            pred_code = REVERSE_TARGET_MAP.get(int(pred_idx), int(pred_idx) + 1)
            label = SEVERITY_NAMES.get(int(pred_idx), f"Severity {pred_code}")
            prob_entry = {}
            if probs is not None:
                prob_entry = {
                    "Fatal": round(float(probs[i][0]), 4),
                    "Serious": round(float(probs[i][1]), 4),
                    "Slight": round(float(probs[i][2]), 4),
                }
            batch_results.append({
                "record_index": i,
                "severity_code": pred_code,
                "severity_label": label,
                "probabilities": prob_entry,
            })

        total_latency_ms = round((time.time() - t0) * 1000, 2)
        return {
            "total_records": len(batch_results),
            "total_latency_ms": total_latency_ms,
            "average_latency_ms": round(total_latency_ms / len(batch_results), 2),
            "results": batch_results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch error: {str(e)}")


@app.get("/metrics", tags=["Operational"])
def get_metrics():
    """
    Retrieves operational telemetry directly from ops.prediction_log.
    """
    try:
        engine = get_db_engine()
        query = text("""
            SELECT 
                COUNT(*) as total_predictions,
                AVG(latency_ms) as avg_latency_ms,
                COUNT(*) FILTER (WHERE predicted_severity_code = 1) as fatal_count,
                COUNT(*) FILTER (WHERE predicted_severity_code = 2) as serious_count,
                COUNT(*) FILTER (WHERE predicted_severity_code = 3) as slight_count
            FROM ops.prediction_log;
        """)
        with engine.connect() as conn:
            row = conn.execute(query).mappings().first()
            if row and row["total_predictions"] > 0:
                return {
                    "total_predictions": row["total_predictions"],
                    "avg_latency_ms": round(float(row["avg_latency_ms"] or 0), 2),
                    "predicted_distribution": {
                        "Fatal": row["fatal_count"],
                        "Serious": row["serious_count"],
                        "Slight": row["slight_count"],
                    }
                }
    except Exception as e:
        logger.warning(f"Telemetry metrics query error: {e}")
        
    return {
        "total_predictions": 0,
        "avg_latency_ms": 0.0,
        "predicted_distribution": {"Fatal": 0, "Serious": 0, "Slight": 0}
    }
