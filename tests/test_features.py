"""
tests/test_features.py
----------------------
Unit and integration tests for feature engineering, data leakage prevention,
and chronological train/validation/test splitting.
"""

import pytest
import numpy as np
import pandas as pd
from src.ml.features import (
    load_model_ready_data,
    split_chronologically,
    extract_features_and_target,
    build_preprocessor,
    TARGET_COL,
)


def test_zero_casualty_leakage():
    """
    CRITICAL TEST: Verifies that no casualty metrics (counts, injuries, severity)
    are present in mart.model_ready features.
    """
    df = load_model_ready_data()
    
    forbidden_substrings = [
        "casualty",
        "injur",
        "fatalities",
        "hospital",
        "breath_test",
    ]

    for col in df.columns:
        if col in ["severity_code", "severity_label"]:
            continue  # Target label is allowed
        for forbidden in forbidden_substrings:
            assert forbidden not in col.lower(), (
                f"LEAKAGE DETECTED: Column '{col}' contains forbidden substring '{forbidden}'."
            )


def test_chronological_ordering():
    """
    Verifies that the chronological split ensures train timestamps precede test timestamps.
    """
    df = load_model_ready_data()
    train_df, val_df, test_df = split_chronologically(df)

    assert train_df["month"].max() <= 8
    assert val_df["month"].min() >= 9
    assert val_df["month"].max() <= 10
    assert test_df["month"].min() >= 11

    # Ensure train and test sets are non-empty
    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0


def test_preprocessor_pipeline_transformation():
    """
    Verifies that the preprocessor ColumnTransformer transforms raw data into 2D float array
    without generating NaNs or dimension mismatches.
    """
    df = load_model_ready_data()
    train_df, _, _ = split_chronologically(df)
    X_train, y_train = extract_features_and_target(train_df)

    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X_train)

    assert isinstance(X_transformed, np.ndarray)
    assert X_transformed.shape[0] == len(X_train)
    assert not np.isnan(X_transformed).any(), "Transformed feature matrix contains NaNs!"
