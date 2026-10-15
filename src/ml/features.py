"""
src/ml/features.py
------------------
Leak-free feature engineering and chronological train/validation/test splitting
for the UK Traffic Accident Severity ML pipeline.

Strict MLOps guarantees:
1. Zero Data Leakage: Strictly excludes casualty counts, casualty injuries, and post-accident events.
2. Chronological Splitting: Evaluates time-series accident data temporally to prevent future leakage into past models.
3. No Train-Serve Skew: Encapsulates all transformations (imputation, scaling, one-hot encoding, cyclical transforms)
   inside a reproducible scikit-learn ColumnTransformer / Pipeline.
"""

import logging
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sqlalchemy import text
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from src.db.connection import get_db_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("ml.features")

# Target definition
TARGET_COL = "severity_code"  # 1: Fatal, 2: Serious, 3: Slight
TARGET_MAP = {1: 0, 2: 1, 3: 2}  # 0-indexed for XGBoost / LightGBM compatibility
REVERSE_TARGET_MAP = {0: 1, 1: 2, 2: 3}
SEVERITY_NAMES = {0: "Fatal", 1: "Serious", 2: "Slight"}

# Feature sets
NUMERICAL_COLS = [
    "speed_limit",
    "num_vehicles",
    "temperature_2m",
    "precipitation",
    "wind_speed_10m",
]

CATEGORICAL_COLS = [
    "road_type",
    "junction_detail",
    "road_surface",
    "light_condition",
    "weather_condition",
    "urban_or_rural",
]

TEMPORAL_COLS = [
    "hour",
    "weekday",
    "month",
    "is_weekend",
    "is_rush_hour",
]


class CyclicalTemporalTransformer(BaseEstimator, TransformerMixin):
    """
    Transforms cyclical time features (hour 0-23, month 1-12) into sine and cosine components.
    Ensures that 23:00 and 00:00 or Dec and Jan are recognized as spatially adjacent in feature space.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        # Assumes input columns: [hour, weekday, month, is_weekend, is_rush_hour]
        hour = X_df.iloc[:, 0].astype(float)
        month = X_df.iloc[:, 2].astype(float)
        
        sin_hour = np.sin(2 * np.pi * hour / 24.0)
        cos_hour = np.cos(2 * np.pi * hour / 24.0)
        sin_month = np.sin(2 * np.pi * month / 12.0)
        cos_month = np.cos(2 * np.pi * month / 12.0)
        
        # Additional flags
        weekday = X_df.iloc[:, 1].astype(float)
        is_weekend = X_df.iloc[:, 3].astype(float)
        is_rush_hour = X_df.iloc[:, 4].astype(float)
        
        return np.column_stack([sin_hour, cos_hour, sin_month, cos_month, weekday, is_weekend, is_rush_hour])


def load_model_ready_data() -> pd.DataFrame:
    """
    Loads all validated, leak-free records from mart.model_ready ordered chronologically.
    """
    engine = get_db_engine()
    query = """
    SELECT 
        accident_id,
        accident_ts,
        hour,
        weekday,
        month,
        is_weekend,
        is_rush_hour,
        road_type,
        speed_limit,
        junction_detail,
        road_surface,
        light_condition,
        weather_condition,
        urban_or_rural,
        grid_cell_id,
        temperature_2m,
        precipitation,
        wind_speed_10m,
        num_vehicles,
        severity_code,
        severity_label
    FROM mart.model_ready
    ORDER BY accident_ts ASC;
    """
    logger.info("Loading model_ready dataset from PostgreSQL...")
    df = pd.read_sql(query, engine)
    logger.info(f"Loaded {len(df):,} records across {df.shape[1]} columns.")
    return df


def split_chronologically(
    df: pd.DataFrame,
    train_end_month: int = 8,   # Jan - Aug (8 months ~67%)
    val_end_month: int = 10,    # Sep - Oct (2 months ~17%)
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs chronological train/validation/test split based on accident_ts calendar month.
    Train: Months 1 to 8 (January - August)
    Validation: Months 9 to 10 (September - October)
    Test: Months 11 to 12 (November - December)
    """
    logger.info(f"Splitting data chronologically: Train (<= Month {train_end_month}), Val (Months {train_end_month+1}-{val_end_month}), Test (> Month {val_end_month})")
    
    train_mask = df["month"] <= train_end_month
    val_mask = (df["month"] > train_end_month) & (df["month"] <= val_end_month)
    test_mask = df["month"] > val_end_month

    df_train = df[train_mask].copy()
    df_val = df[val_mask].copy()
    df_test = df[test_mask].copy()

    logger.info(f"Split sizes - Train: {len(df_train):,} ({len(df_train)/len(df):.1%}), "
                f"Val: {len(df_val):,} ({len(df_val)/len(df):.1%}), "
                f"Test: {len(df_test):,} ({len(df_test)/len(df):.1%})")

    return df_train, df_val, df_test


def build_preprocessor() -> ColumnTransformer:
    """
    Constructs the scikit-learn ColumnTransformer pipeline for features.
    Encapsulates:
    - Numerical: Median imputation + Standard Scaling
    - Categorical: Most frequent imputation + OneHotEncoding (ignore unknown categories)
    - Temporal: Cyclical sin/cos encoding for hour and month + passthrough flags
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    temp_pipeline = Pipeline([
        ("cyclical", CyclicalTemporalTransformer()),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_COLS),
            ("cat", cat_pipeline, CATEGORICAL_COLS),
            ("temp", temp_pipeline, TEMPORAL_COLS),
        ],
        remainder="drop",
    )

    return preprocessor


def extract_features_and_target(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Separates the feature matrix X from the multiclass target y.
    Converts target from {1: Fatal, 2: Serious, 3: Slight} to {0: Fatal, 1: Serious, 2: Slight}.
    """
    feature_cols = NUMERICAL_COLS + CATEGORICAL_COLS + TEMPORAL_COLS
    X = df[feature_cols].copy()
    y = df[TARGET_COL].map(TARGET_MAP).values
    return X, y


if __name__ == "__main__":
    df = load_model_ready_data()
    train_df, val_df, test_df = split_chronologically(df)
    
    X_train, y_train = extract_features_and_target(train_df)
    X_val, y_val = extract_features_and_target(val_df)
    X_test, y_test = extract_features_and_target(test_df)
    
    preprocessor = build_preprocessor()
    X_train_trans = preprocessor.fit_transform(X_train)
    X_val_trans = preprocessor.transform(X_val)
    X_test_trans = preprocessor.transform(X_test)
    
    print("\nPreprocessed Feature Shapes:")
    print(f"X_train: {X_train_trans.shape}, y_train: {y_train.shape}")
    print(f"X_val:   {X_val_trans.shape}, y_val:   {y_val.shape}")
    print(f"X_test:  {X_test_trans.shape}, y_test:  {y_test.shape}")
    
    # Class distribution in train
    unique, counts = np.unique(y_train, return_counts=True)
    print("\nTrain Class Distribution:")
    for u, c in zip(unique, counts):
        print(f"Class {u} ({SEVERITY_NAMES[u]}): {c:,} ({c/len(y_train):.2%})")
