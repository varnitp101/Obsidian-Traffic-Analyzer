-- ============================================================================
-- UK Traffic Accident Severity Platform - Master Database Schema (PostgreSQL)
-- Implements the 5-layer architecture: raw, staging, clean, mart, ops
-- ============================================================================

-- Enable PostGIS if extension is available (safe fallback)
DO $$
BEGIN
    CREATE EXTENSION IF NOT EXISTS postgis;
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'PostGIS extension not available; proceeding with standard lat/long columns.';
END $$;

-- ----------------------------------------------------------------------------
-- 1. SCHEMAS INITIALIZATION
-- ----------------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS clean;
CREATE SCHEMA IF NOT EXISTS mart;
CREATE SCHEMA IF NOT EXISTS ops;

-- ----------------------------------------------------------------------------
-- 2. OPS SCHEMA: Pipeline Telemetry, Logging & Data Quality
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ops.ingestion_log (
    run_id VARCHAR(64) PRIMARY KEY,
    source_name VARCHAR(64) NOT NULL,
    extraction_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) NOT NULL, -- 'success', 'failed', 'cached'
    records_extracted INTEGER DEFAULT 0,
    file_path TEXT,
    file_size_bytes BIGINT,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS ops.rejected_records (
    rejection_id BIGSERIAL PRIMARY KEY,
    run_id VARCHAR(64),
    source_table VARCHAR(64) NOT NULL,
    record_id VARCHAR(64),
    rule_name VARCHAR(128) NOT NULL,
    reason TEXT NOT NULL,
    raw_payload JSONB,
    rejected_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ops.dq_results (
    dq_run_id BIGSERIAL PRIMARY KEY,
    run_id VARCHAR(64),
    layer_name VARCHAR(32) NOT NULL,
    table_name VARCHAR(64) NOT NULL,
    rows_in INTEGER NOT NULL,
    rows_passed INTEGER NOT NULL,
    rows_rejected INTEGER NOT NULL,
    dq_metric_details JSONB,
    evaluated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ops.prediction_log (
    prediction_id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    model_version VARCHAR(32) NOT NULL,
    inputs JSONB NOT NULL,
    predicted_severity_code SMALLINT NOT NULL,
    predicted_severity_label VARCHAR(16) NOT NULL,
    class_probabilities JSONB NOT NULL,
    latency_ms DOUBLE PRECISION NOT NULL,
    error_flag BOOLEAN DEFAULT FALSE,
    error_message TEXT
);

-- ----------------------------------------------------------------------------
-- 3. STAGING SCHEMA: Mirroring Raw Files with Minimal Casts
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS staging.collision (
    collision_index VARCHAR(32) PRIMARY KEY,
    collision_year INTEGER,
    collision_ref_no VARCHAR(32),
    location_easting_osgr VARCHAR(32),
    location_northing_osgr VARCHAR(32),
    longitude VARCHAR(32),
    latitude VARCHAR(32),
    police_force VARCHAR(32),
    collision_severity VARCHAR(16),
    number_of_vehicles VARCHAR(16),
    number_of_casualties VARCHAR(16),
    date VARCHAR(32),
    day_of_week VARCHAR(16),
    time VARCHAR(16),
    local_authority_district VARCHAR(32),
    local_authority_ons_district VARCHAR(32),
    first_road_class VARCHAR(16),
    first_road_number VARCHAR(16),
    road_type VARCHAR(16),
    speed_limit VARCHAR(16),
    junction_detail VARCHAR(16),
    junction_control VARCHAR(16),
    second_road_class VARCHAR(16),
    second_road_number VARCHAR(16),
    pedestrian_crossing VARCHAR(16),
    light_conditions VARCHAR(16),
    weather_conditions VARCHAR(16),
    road_surface_conditions VARCHAR(16),
    special_conditions_at_site VARCHAR(16),
    carriageway_hazards VARCHAR(16),
    urban_or_rural_area VARCHAR(16),
    did_police_officer_attend_scene_of_accident VARCHAR(16),
    trunk_road_flag VARCHAR(16),
    lsoa_of_accident_location VARCHAR(32)
);

CREATE TABLE IF NOT EXISTS staging.vehicle (
    collision_index VARCHAR(32),
    collision_year INTEGER,
    collision_ref_no VARCHAR(32),
    vehicle_reference VARCHAR(16),
    vehicle_type VARCHAR(16),
    towing_and_articulation VARCHAR(16),
    vehicle_manoeuvre VARCHAR(16),
    vehicle_direction_from VARCHAR(16),
    vehicle_direction_to VARCHAR(16),
    vehicle_location_restricted_lane VARCHAR(16),
    junction_location VARCHAR(16),
    skidding_and_overturning VARCHAR(16),
    hit_object_in_carriageway VARCHAR(16),
    vehicle_leaving_carriageway VARCHAR(16),
    hit_object_off_carriageway VARCHAR(16),
    first_point_of_impact VARCHAR(16),
    vehicle_left_hand_drive VARCHAR(16),
    journey_purpose_of_driver VARCHAR(16),
    sex_of_driver VARCHAR(16),
    age_of_driver VARCHAR(16),
    age_band_of_driver VARCHAR(16),
    engine_capacity_cc VARCHAR(16),
    propulsion_code VARCHAR(16),
    age_of_vehicle VARCHAR(16),
    generic_make_model VARCHAR(64),
    driver_imd_decile VARCHAR(16),
    driver_home_area_type VARCHAR(16)
);

CREATE TABLE IF NOT EXISTS staging.casualty (
    collision_index VARCHAR(32),
    collision_year INTEGER,
    collision_ref_no VARCHAR(32),
    vehicle_reference VARCHAR(16),
    casualty_reference VARCHAR(16),
    casualty_class VARCHAR(16),
    sex_of_casualty VARCHAR(16),
    age_of_casualty VARCHAR(16),
    age_band_of_casualty VARCHAR(16),
    casualty_severity VARCHAR(16),
    pedestrian_location VARCHAR(16),
    pedestrian_movement VARCHAR(16),
    car_passenger VARCHAR(16),
    bus_or_coach_passenger VARCHAR(16),
    pedestrian_road_maintenance_worker VARCHAR(16),
    casualty_type VARCHAR(16),
    casualty_home_area_type VARCHAR(16),
    casualty_imd_decile VARCHAR(16)
);

CREATE TABLE IF NOT EXISTS staging.weather_hourly (
    grid_cell_id VARCHAR(32),
    grid_latitude DOUBLE PRECISION,
    grid_longitude DOUBLE PRECISION,
    weather_timestamp TIMESTAMP WITHOUT TIME ZONE,
    temperature_2m DOUBLE PRECISION,
    precipitation DOUBLE PRECISION,
    wind_speed_10m DOUBLE PRECISION,
    weather_code INTEGER,
    cloud_cover DOUBLE PRECISION,
    PRIMARY KEY (grid_cell_id, weather_timestamp)
);

-- ----------------------------------------------------------------------------
-- 4. CLEAN SCHEMA: Deduplicated, Standardized & Decoded Records
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS clean.accident (
    accident_id VARCHAR(32) PRIMARY KEY,
    accident_ts TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    grid_cell_id VARCHAR(32) NOT NULL,
    severity_code SMALLINT NOT NULL CHECK (severity_code IN (1, 2, 3)),
    severity_label VARCHAR(16) NOT NULL,
    num_vehicles SMALLINT NOT NULL DEFAULT 1,
    num_casualties SMALLINT NOT NULL DEFAULT 1,
    day_of_week SMALLINT NOT NULL,
    road_type_code SMALLINT,
    road_type_label VARCHAR(64),
    speed_limit SMALLINT,
    junction_detail_code SMALLINT,
    junction_detail_label VARCHAR(64),
    light_code SMALLINT,
    light_label VARCHAR(64),
    weather_code SMALLINT,
    weather_label VARCHAR(64),
    road_surface_code SMALLINT,
    road_surface_label VARCHAR(64),
    urban_rural_code SMALLINT,
    urban_rural_label VARCHAR(16),
    police_force_code SMALLINT,
    local_authority_code VARCHAR(32),
    clean_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS clean.vehicle (
    vehicle_id BIGSERIAL PRIMARY KEY,
    accident_id VARCHAR(32) NOT NULL REFERENCES clean.accident(accident_id) ON DELETE CASCADE,
    vehicle_ref SMALLINT NOT NULL,
    vehicle_type_code SMALLINT,
    vehicle_type_label VARCHAR(64),
    manoeuvre_code SMALLINT,
    manoeuvre_label VARCHAR(64),
    sex_of_driver_code SMALLINT,
    sex_of_driver_label VARCHAR(16),
    age_of_driver SMALLINT,
    age_band_driver_code SMALLINT,
    engine_capacity_cc INTEGER,
    age_of_vehicle SMALLINT
);

CREATE TABLE IF NOT EXISTS clean.casualty (
    casualty_id BIGSERIAL PRIMARY KEY,
    accident_id VARCHAR(32) NOT NULL REFERENCES clean.accident(accident_id) ON DELETE CASCADE,
    vehicle_ref SMALLINT,
    casualty_ref SMALLINT NOT NULL,
    casualty_class_code SMALLINT,
    casualty_class_label VARCHAR(32),
    sex_of_casualty_code SMALLINT,
    sex_of_casualty_label VARCHAR(16),
    age_of_casualty SMALLINT,
    casualty_severity_code SMALLINT CHECK (casualty_severity_code IN (1, 2, 3)),
    casualty_severity_label VARCHAR(16)
);

-- ----------------------------------------------------------------------------
-- 5. MART SCHEMA: Dimensional Model (Star Schema) & Model-Ready Table
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS mart.dim_date (
    date_key INTEGER PRIMARY KEY,           -- YYYYMMDD
    full_date DATE NOT NULL,
    year SMALLINT NOT NULL,
    month SMALLINT NOT NULL,
    month_name VARCHAR(16) NOT NULL,
    day SMALLINT NOT NULL,
    day_of_week SMALLINT NOT NULL,
    weekday_name VARCHAR(16) NOT NULL,
    hour SMALLINT NOT NULL,
    is_weekend BOOLEAN NOT NULL,
    is_rush_hour BOOLEAN NOT NULL
);

CREATE TABLE IF NOT EXISTS mart.dim_road (
    road_key SERIAL PRIMARY KEY,
    road_type VARCHAR(64) NOT NULL,
    speed_limit SMALLINT NOT NULL,
    junction_detail VARCHAR(64) NOT NULL,
    road_surface VARCHAR(64) NOT NULL,
    UNIQUE (road_type, speed_limit, junction_detail, road_surface)
);

CREATE TABLE IF NOT EXISTS mart.dim_weather (
    weather_key SERIAL PRIMARY KEY,
    weather_condition VARCHAR(64) NOT NULL,
    light_condition VARCHAR(64) NOT NULL,
    temp_category VARCHAR(32) NOT NULL,     -- 'Freezing', 'Cold', 'Mild', 'Warm', 'Hot'
    rain_category VARCHAR(32) NOT NULL,     -- 'None', 'Light', 'Moderate', 'Heavy'
    wind_category VARCHAR(32) NOT NULL,     -- 'Calm', 'Breeze', 'Gale'
    UNIQUE (weather_condition, light_condition, temp_category, rain_category, wind_category)
);

CREATE TABLE IF NOT EXISTS mart.dim_location (
    location_key SERIAL PRIMARY KEY,
    grid_cell_id VARCHAR(32) NOT NULL,
    grid_latitude DOUBLE PRECISION NOT NULL,
    grid_longitude DOUBLE PRECISION NOT NULL,
    local_authority VARCHAR(64),
    police_force VARCHAR(64),
    urban_rural VARCHAR(16) NOT NULL,
    UNIQUE (grid_cell_id, local_authority, police_force, urban_rural)
);

CREATE TABLE IF NOT EXISTS mart.dim_severity (
    severity_key SMALLINT PRIMARY KEY,
    severity_label VARCHAR(16) NOT NULL,
    description TEXT
);

INSERT INTO mart.dim_severity (severity_key, severity_label, description)
VALUES 
    (1, 'Fatal', 'Accidents where at least one person suffered fatal injuries within 30 days of the collision'),
    (2, 'Serious', 'Accidents where at least one person suffered severe injuries requiring hospitalization'),
    (3, 'Slight', 'Accidents with minor cuts, bruises, shock, or no visible external injuries')
ON CONFLICT (severity_key) DO NOTHING;

-- Fact Table
CREATE TABLE IF NOT EXISTS mart.fact_accident (
    accident_id VARCHAR(32) PRIMARY KEY,
    date_key INTEGER NOT NULL REFERENCES mart.dim_date(date_key),
    road_key INTEGER NOT NULL REFERENCES mart.dim_road(road_key),
    weather_key INTEGER NOT NULL REFERENCES mart.dim_weather(weather_key),
    location_key INTEGER NOT NULL REFERENCES mart.dim_location(location_key),
    severity_key SMALLINT NOT NULL REFERENCES mart.dim_severity(severity_key),
    
    -- Measures
    num_vehicles SMALLINT NOT NULL,
    num_casualties SMALLINT NOT NULL,
    temperature_2m DOUBLE PRECISION,
    precipitation DOUBLE PRECISION,
    wind_speed_10m DOUBLE PRECISION,
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL
);

-- Model-Ready Flat Table (Strictly known-at-time-of-accident features, NO LEAKAGE)
CREATE TABLE IF NOT EXISTS mart.model_ready (
    accident_id VARCHAR(32) PRIMARY KEY,
    accident_ts TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    hour SMALLINT NOT NULL,
    weekday SMALLINT NOT NULL,
    month SMALLINT NOT NULL,
    is_weekend BOOLEAN NOT NULL,
    is_rush_hour BOOLEAN NOT NULL,
    road_type VARCHAR(64) NOT NULL,
    speed_limit SMALLINT NOT NULL,
    junction_detail VARCHAR(64) NOT NULL,
    road_surface VARCHAR(64) NOT NULL,
    light_condition VARCHAR(64) NOT NULL,
    weather_condition VARCHAR(64) NOT NULL,
    urban_or_rural VARCHAR(16) NOT NULL,
    grid_cell_id VARCHAR(32) NOT NULL,
    temperature_2m DOUBLE PRECISION,
    precipitation DOUBLE PRECISION,
    wind_speed_10m DOUBLE PRECISION,
    num_vehicles SMALLINT NOT NULL,
    
    -- TARGET (Multiclass: 1=Fatal, 2=Serious, 3=Slight)
    severity_code SMALLINT NOT NULL CHECK (severity_code IN (1, 2, 3)),
    severity_label VARCHAR(16) NOT NULL
);

-- ----------------------------------------------------------------------------
-- 6. PERFORMANCE INDEXES
-- ----------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_fact_date ON mart.fact_accident(date_key);
CREATE INDEX IF NOT EXISTS idx_fact_severity ON mart.fact_accident(severity_key);
CREATE INDEX IF NOT EXISTS idx_fact_location ON mart.fact_accident(location_key);
CREATE INDEX IF NOT EXISTS idx_clean_accident_ts ON clean.accident(accident_ts);
CREATE INDEX IF NOT EXISTS idx_clean_accident_grid ON clean.accident(grid_cell_id);
CREATE INDEX IF NOT EXISTS idx_model_ready_ts ON mart.model_ready(accident_ts);
CREATE INDEX IF NOT EXISTS idx_model_ready_target ON mart.model_ready(severity_code);
