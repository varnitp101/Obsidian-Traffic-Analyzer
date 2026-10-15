"""
tests/test_etl.py
-----------------
Unit tests for data validation, cleaning, and rejection logic.
"""

import pytest
import pandas as pd
from src.etl.validate import (
    validate_collisions,
    validate_vehicles,
    validate_casualties,
)


def test_collision_coordinate_validation():
    """Tests that out-of-bounds and null coordinates are identified for rejection."""
    df_raw = pd.DataFrame([
        {
            "collision_index": "VALID_01",
            "collision_year": 2022,
            "latitude": "51.5074",
            "longitude": "-0.1278",
            "collision_severity": "3",
            "number_of_vehicles": "2",
            "number_of_casualties": "1",
            "date": "15/05/2022",
            "time": "14:30",
        },
        {
            "collision_index": "INVALID_NULL_COORDS",
            "collision_year": 2022,
            "latitude": None,
            "longitude": "-0.1278",
            "collision_severity": "3",
            "number_of_vehicles": "1",
            "number_of_casualties": "1",
            "date": "15/05/2022",
            "time": "14:30",
        },
        {
            "collision_index": "INVALID_OUT_OF_BOUNDS",
            "collision_year": 2022,
            "latitude": "25.0000",  # Out of UK bounds
            "longitude": "75.0000",
            "collision_severity": "1",
            "number_of_vehicles": "1",
            "number_of_casualties": "1",
            "date": "15/05/2022",
            "time": "14:30",
        },
    ])

    passed, rejections = validate_collisions(df_raw, run_id="test_run")
    assert len(passed) == 1
    assert passed.iloc[0]["collision_index"] == "VALID_01"
    assert len(rejections) == 2

    rule_names = rejections["rule_name"].tolist()
    assert "NULL_COORDINATES" in rule_names
    assert "OUT_OF_BOUNDS_COORDINATES" in rule_names


def test_orphan_vehicle_rejection():
    """Tests that vehicles referencing non-existent collision parents are rejected."""
    valid_ids = {"COLL_001", "COLL_002"}
    df_vehicles = pd.DataFrame([
        {"collision_index": "COLL_001", "vehicle_reference": "1", "vehicle_type": "9"},
        {"collision_index": "ORPHAN_999", "vehicle_reference": "1", "vehicle_type": "9"},
    ])

    passed, rejections = validate_vehicles(df_vehicles, valid_collision_ids=valid_ids, run_id="test_run")
    assert len(passed) == 1
    assert passed.iloc[0]["collision_index"] == "COLL_001"
    assert len(rejections) == 1
    assert rejections.iloc[0]["rule_name"] == "ORPHAN_VEHICLE"
