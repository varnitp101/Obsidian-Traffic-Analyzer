"""
tests/test_api.py
-----------------
Integration tests for FastAPI inference microservice.
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health_endpoint():
    """Tests that the health endpoint returns 200 with operational metadata."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "uptime_seconds" in data
    assert data["model_loaded"] is True


def test_model_info_endpoint():
    """Tests model metadata and feature schema endpoint."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "target_mapping" in data
    assert "input_features" in data
    assert "zero_leakage_guarantee" in data


def test_single_prediction_endpoint():
    """Tests /predict with a realistic collision payload."""
    payload = {
        "speed_limit": 60.0,
        "num_vehicles": 2,
        "temperature_2m": 3.5,
        "precipitation": 1.2,
        "wind_speed_10m": 25.0,
        "road_type": "Single carriageway",
        "junction_detail": "T or staggered junction",
        "road_surface": "Wet or damp",
        "light_condition": "Darkness - lights unlit",
        "weather_condition": "Raining + high winds",
        "urban_or_rural": "Rural",
        "hour": 22,
        "weekday": 6,
        "month": 11,
        "is_weekend": 1,
        "is_rush_hour": 0,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert "prediction_id" in data
    assert data["severity_code"] in [1, 2, 3]
    assert data["severity_label"] in ["Fatal", "Serious", "Slight"]
    assert "probabilities" in data
    assert "Fatal" in data["probabilities"]
    assert "Serious" in data["probabilities"]
    assert "Slight" in data["probabilities"]
    assert data["latency_ms"] > 0


def test_batch_prediction_endpoint():
    """Tests /predict/batch endpoint with multiple collision records."""
    sample_accident = {
        "speed_limit": 30.0,
        "num_vehicles": 1,
        "temperature_2m": 15.0,
        "precipitation": 0.0,
        "wind_speed_10m": 10.0,
        "road_type": "Single carriageway",
        "junction_detail": "Not at junction or within 20 metres",
        "road_surface": "Dry",
        "light_condition": "Daylight",
        "weather_condition": "Fine no high winds",
        "urban_or_rural": "Urban",
        "hour": 14,
        "weekday": 3,
        "month": 6,
        "is_weekend": 0,
        "is_rush_hour": 0,
    }

    payload = {"accidents": [sample_accident, sample_accident]}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_records"] == 2
    assert len(data["results"]) == 2
