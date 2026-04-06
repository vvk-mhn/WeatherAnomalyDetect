import os
import pandas as pd
from datetime import datetime, timedelta
from fmiopendata.wfs import download_stored_query

# CONFIG
START_TIME = datetime(2023, 1, 1)
END_TIME   = datetime(2023, 2, 1)

PARAMS = [
    "t2m",          # temperature
    "rh",           # humidity
    "ws_10min",     # wind speed
    "r_1h"          # precipitation
]

# Map fmi short names to descriptive names
PARAM_NAMES = [
    "Air temperature",
    "Relative humidity",
    "Wind speed",
    "Precipitation amount"
]

MIN_DATA_RATIO = 0.8   # keep stations with ≥80% data coverage
OUTPUT_DIR = "./fmi_data/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# FETCH DATA
station_dfs = {}
current_start = START_TIME

while current_start < END_TIME:
    current_end = min(current_start + timedelta(days=6), END_TIME)
    print(f"  Fetching {current_start.date()} to {current_end.date()}...")
    
    try:
        query = download_stored_query(
            "fmi::observations::weather::multipointcoverage",
            args=[
                "bbox=18,55,35,75",
                f"starttime={current_start.isoformat()}Z",
                f"endtime={current_end.isoformat()}Z",
                "timestep=60",  # hourly data
                "parameters=" + ",".join(PARAMS),
                "timeseries=True"
            ]
        )
        
        for station_id, station_data in query.data.items():
            times = station_data.get("times")
            if not times:
                continue

            # Build dictionary for dataframe
            df_dict = {}
            for param in PARAM_NAMES:
                if param in station_data and "values" in station_data[param]:
                    df_dict[param] = station_data[param]["values"]
                else:
                    df_dict[param] = [None] * len(times)

            df = pd.DataFrame(df_dict, index=times)
            
            if station_id not in station_dfs:
                station_dfs[station_id] = []
            station_dfs[station_id].append(df)
            
    except Exception as e:
        print(f"  Failed to fetch chunk: {e}")
        
    current_start = current_end

# PARSE DATA
print("Parsing data...")

valid_stations = []

for station_id, dfs in station_dfs.items():
    try:
        if not dfs:
            continue

        # Concatenate chunks for the station
        df = pd.concat(dfs)

        df.index = pd.to_datetime(df.index)
        
        # Convert columns to numeric, coercing errors to NaN
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Sort
        df = df.sort_index()

        # CLEANING
        df = df.dropna(axis=1, how="all")

        if df.empty:
            continue

        # Compute data coverage
        total_points = len(df)
        valid_points = df.dropna().shape[0]
        ratio = valid_points / total_points

        if ratio < MIN_DATA_RATIO:
            continue

        # RESAMPLE (hourly grid)
        df = df.resample("1h").mean()

        # Fill small gaps
        df = df.interpolate(limit=3)
        df = df.dropna()

        if len(df) < 24 * 7:  # at least 1 week data
            continue

        # SAVE
        # Clean station name to be valid filename
        filename_safe = "".join(c if c.isalnum() else "_" for c in station_id)
        filename = os.path.join(OUTPUT_DIR, f"{filename_safe}.csv")
        df.to_csv(filename)

        valid_stations.append(station_id)

        print(f"Saved: {station_id} | Samples: {len(df)}")

    except Exception as e:
        print(f"Skipping {station_id}: {e}")

print(f"Total valid stations: {len(valid_stations)}")
