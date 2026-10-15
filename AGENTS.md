# AGENTS.md - Context Preservation & Operating Protocol

## 1. Project Overview & Mission
This project is an end-to-end Data Engineering and MLOps platform for **UK Traffic Accident Severity Analytics and Prediction** (based on Department for Transport STATS19 data and Open-Meteo historical weather data).

The architecture consists of two interconnected phases:
1. **Part 1 (Data Engineering Foundation)**:
   - Ingestion of raw STATS19 CSVs (collisions, vehicles, casualties) + Open-Meteo historical weather.
   - Idempotent ETL pipeline (Extract, Clean, Validate, Join).
   - PostgreSQL/PostGIS database structured with strict schema separation (`raw`, `staging`, `clean`, `mart`, `ops`).
   - Dimensional Star Schema and a consolidated `mart.model_ready` table.
   - Apache Airflow orchestration.
   - Interactive Streamlit analytics dashboard (5+ analytical views + pipeline health).
2. **Part 2 (MLOps Extension)**:
   - Leak-free feature pipeline (excluding post-accident casualty metrics).
   - Chronological train/validation/test splitting.
   - Imbalance-aware multiclass modeling (Fatal vs Serious vs Slight) with focus on minority recall.
   - MLflow experiment tracking & model registry (with PostgreSQL backend).
   - Unified scikit-learn preprocessing + model pipeline (preventing train-serve skew).
   - Containerized FastAPI inference microservice (`/predict`, `/health`, `/model-info`, `/metrics`).
   - Prediction integration into Streamlit.
   - Continuous monitoring (data quality, PSI/KS feature drift, class drift, latency, false negatives) & automated retraining triggers.

---

## 2. Strict Architectural & Implementation Rules
Any agent contributing to this repository **MUST** adhere to the following principles:

1. **Weather Ingestion Design**:
   - **DO NOT** call the weather API per accident (100k+ accidents will breach rate limits and crash).
   - **DO** group accidents into spatial grid cells (e.g. `0.25°` lat/lon, ~25 km), request hourly historical weather for distinct grid centers in batch date ranges, store in `staging.weather_hourly`, and join on `grid_cell_id` + floored accident hour.
2. **Database Integrity & Schema Separation**:
   - **NEVER** edit tables manually. All mutations originate from idempotent code.
   - Strict 5-schema architecture:
     - `raw`: Raw files metadata and optional text dumps.
     - `staging`: Raw CSV mirrors with minimal typing.
     - `clean`: Deduplicated, validated, standardized records with lookup translations.
     - `mart`: Dimensional model (fact + dimensions) and `mart.model_ready`.
     - `ops`: Operational logs (`ingestion_log`, `rejected_records`, `dq_results`, `prediction_log`).
3. **Data Quality & Rejection Logging**:
   - Any record failing validation rules (null coordinates, invalid dates, out-of-UK bounds, invalid severity codes) must NOT be silently dropped.
   - Rejected records are logged to `ops.rejected_records` and `data/rejected/*.csv` with `source_table`, `record_id`, `rule_name`, `reason`, and `timestamp`.
4. **Data Leakage Prohibition**:
   - The ML model predicts accident severity (Fatal / Serious / Slight) based **ONLY** on features known at the time of the collision.
   - Exclude casualty counts, casualty injuries, and post-event actions from features.
5. **No Train-Serve Skew**:
   - Feature engineering and preprocessing must be encapsulated inside a single scikit-learn `Pipeline` (with `ColumnTransformer`), saved and versioned in MLflow. The API takes raw input fields directly.
6. **Chronological Splitting**:
   - Never use random train/test splits for time-series / temporal accident data.
   - Always split chronologically (e.g., historical years for training, subsequent period for validation, latest period for testing). Resampling (e.g. SMOTE) must ONLY be applied to the training set.

---

## 3. Directory Structure
```
TrafficAccidentMLOps/
├── .env.example
├── .gitignore
├── AGENTS.md
├── IMPLEMENTATION.md
├── README.md
├── docker-compose.yml
├── data/
│   ├── raw/
│   │   ├── stats19/
│   │   └── weather/
│   └── rejected/
├── dags/
│   └── accident_pipeline_dag.py
├── src/
│   ├── ingestion/
│   │   ├── download_stats19.py
│   │   └── fetch_weather.py
│   ├── etl/
│   │   ├── clean.py
│   │   ├── validate.py
│   │   ├── join_weather.py
│   │   └── build_mart.py
│   ├── db/
│   │   ├── schema.sql
│   │   ├── connection.py
│   │   └── load.py
│   ├── ml/
│   │   ├── features.py
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── monitoring/
│   │   ├── drift.py
│   │   └── performance.py
│   ├── api/
│   │   ├── main.py
│   │   └── Dockerfile
│   └── dashboard/
│       └── app.py
├── docs/
│   ├── architecture_diagram.png
│   ├── data_dictionary.md
│   └── report.md
├── notebooks/
│   └── 01_data_exploration.ipynb
└── tests/
    ├── test_etl.py
    ├── test_api.py
    └── test_features.py
```

---

## 4. Build Sequence & Progress Tracker

- [x] **Step 1: Data Acquisition & Exploratory Analysis**
  - Download STATS19 CSVs (collisions, vehicles, casualties for target year).
  - Exploratory data analysis, column profiling, lookup code mapping.
- [x] **Step 2: Database Schemas & Storage Layer**
  - Define PostgreSQL DDL (`raw`, `staging`, `clean`, `mart`, `ops`).
  - Create database connection & loading utilities (`src/db/`).
- [x] **Step 3: Ingestion Scripts & Raw Layer with Logging**
  - Implement `download_stats19.py` and `fetch_weather.py` with exponential backoff & retry.
  - Populate `ops.ingestion_log`.
- [x] **Step 4: Cleaning & Validation (Rejected Records Log)**
  - Implement cleaning rules & rejection filters in `src/etl/validate.py` & `clean.py`.
  - Populate `ops.rejected_records` and `ops.dq_results`.
- [x] **Step 5: Weather Ingestion & Grid Spatial Join**
  - Grid cell partitioning (`0.25°`), hourly batch API fetch, weather-to-accident join.
- [x] **Step 6: Dimensional Model & `mart.model_ready`**
  - Build star schema (`fact_accident`, `dim_date`, `dim_road`, `dim_weather`, `dim_location`, `dim_severity`).
  - Generate the wide, leak-free `mart.model_ready` table.
- [x] **Step 7: Apache Airflow DAG Orchestration**
  - Build `accident_pipeline_dag.py` with end-to-end task dependencies.
- [x] **Step 8: Streamlit Analytics Dashboard**
  - Implement 5+ interactive views + data quality / pipeline health monitor.
- [x] **Step 9: Part 1 Documentation, DDLs, and Verification**
- [x] **Step 10: ML Feature Engineering & Chronological Splitting**
  - Build `src/ml/features.py`, strict time-based splits, class imbalance treatment.
- [x] **Step 11: Model Training, Evaluation & MLflow Tracking**
  - Train baseline, Random Forest, LightGBM/XGBoost.
  - Track metrics (macro-F1, per-class recall) and register model in MLflow Model Registry.
- [x] **Step 12: Production Inference Service (FastAPI) & Dockerization**
  - Expose `/predict`, `/health`, `/model-info`, `/metrics` using unified sklearn Pipeline.
- [x] **Step 13: Prediction Interface in Streamlit**
- [x] **Step 14: Continuous Monitoring (Drift, Latency, False Negatives) & Retraining Policy**
- [x] **Step 15: Full Docker Compose Integration, Final Report & Submission Package**
