"""
Data Exploration & Schema Profiling Script for STATS19 UK Accident Dataset.
Analyzes 2022 Collisions, Vehicles, and Casualties.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np

def run_exploration():
    base_dir = Path("data/raw/stats19/2022")
    col_file = base_dir / "dft-road-casualty-statistics-collision-2022.csv"
    veh_file = base_dir / "dft-road-casualty-statistics-vehicle-2022.csv"
    cas_file = base_dir / "dft-road-casualty-statistics-casualty-2022.csv"

    print("=" * 70)
    print("STEP 1: EXPLORATORY DATA ANALYSIS & PROFILING (STATS19 2022)")
    print("=" * 70)

    # 1. Collisions
    print("\n[1] Profiling Collisions...")
    df_col = pd.read_csv(col_file, low_memory=False)
    print(f"Total collision records: {len(df_col):,}")
    print(f"Total columns: {len(df_col.columns)}")

    # Severity distribution
    sev_counts = df_col['collision_severity'].value_counts().sort_index()
    sev_map = {1: 'Fatal', 2: 'Serious', 3: 'Slight'}
    print("\nCollision Severity Breakdown:")
    for code, count in sev_counts.items():
        pct = (count / len(df_col)) * 100
        print(f"  Class {code} ({sev_map.get(code, 'Unknown')}): {count:,} ({pct:.2f}%)")

    # Coordinates & UK Bounds validation
    null_coords = df_col['latitude'].isnull() | df_col['longitude'].isnull()
    print(f"\nMissing Coordinates: {null_coords.sum():,} ({null_coords.mean()*100:.3f}%)")

    # UK roughly lat 49 to 61, lon -9 to 2
    valid_lat = (df_col['latitude'] >= 49.0) & (df_col['latitude'] <= 61.0)
    valid_lon = (df_col['longitude'] >= -9.0) & (df_col['longitude'] <= 2.0)
    valid_uk_bounds = valid_lat & valid_lon & (~null_coords)
    invalid_bounds = (~valid_uk_bounds).sum()
    print(f"Coordinates outside UK bounding box (49-61 lat, -9-2 lon): {invalid_bounds:,}")

    # Duplicates
    dup_ids = df_col['collision_index'].duplicated().sum()
    print(f"Duplicate collision_index entries: {dup_ids:,}")

    # Temporal breakdown
    df_col['dt'] = pd.to_datetime(df_col['date'], format='%d/%m/%Y', errors='coerce')
    missing_dates = df_col['dt'].isnull().sum()
    print(f"Missing or unparseable dates: {missing_dates:,}")
    if missing_dates == 0:
        print(f"Date range: {df_col['dt'].min().strftime('%Y-%m-%d')} to {df_col['dt'].max().strftime('%Y-%m-%d')}")

    # 2. Vehicles
    print("\n" + "-" * 50)
    print("[2] Profiling Vehicles...")
    df_veh = pd.read_csv(veh_file, low_memory=False)
    print(f"Total vehicle records: {len(df_veh):,}")
    # Orphan vehicles
    orphan_veh = (~df_veh['collision_index'].isin(df_col['collision_index'])).sum()
    print(f"Orphan vehicle records (no matching collision): {orphan_veh:,}")

    # 3. Casualties
    print("\n" + "-" * 50)
    print("[3] Profiling Casualties...")
    df_cas = pd.read_csv(cas_file, low_memory=False)
    print(f"Total casualty records: {len(df_cas):,}")
    # Orphan casualties
    orphan_cas = (~df_cas['collision_index'].isin(df_col['collision_index'])).sum()
    print(f"Orphan casualty records (no matching collision): {orphan_cas:,}")

    # Casualty severity alignment
    cas_sev = df_cas['casualty_severity'].value_counts().sort_index()
    print("Casualty Severity Breakdown:")
    for code, count in cas_sev.items():
        pct = (count / len(df_cas)) * 100
        print(f"  Class {code} ({sev_map.get(code, 'Unknown')}): {count:,} ({pct:.2f}%)")

    # Grid Cell Clustering simulation for Weather API
    print("\n" + "-" * 50)
    print("[4] Weather Spatial Grid Simulation (0.25 degree resolution)...")
    valid_coords_df = df_col.loc[valid_uk_bounds, ['latitude', 'longitude']].copy()
    valid_coords_df['lat_grid'] = (valid_coords_df['latitude'] / 0.25).round() * 0.25
    valid_coords_df['lon_grid'] = (valid_coords_df['longitude'] / 0.25).round() * 0.25
    unique_cells = valid_coords_df.drop_duplicates(subset=['lat_grid', 'lon_grid'])
    print(f"Total distinct weather grid cells across {len(valid_coords_df):,} accidents: {len(unique_cells)}")
    print(f"-> Reduces API requests from 106,004 individual calls down to just {len(unique_cells)} batch requests!")

    print("\n" + "=" * 70)
    print("EXPLORATION COMPLETE - Key findings logged.")
    print("=" * 70)

if __name__ == "__main__":
    run_exploration()
