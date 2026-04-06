import os
import glob
import pandas as pd

def load_fmi_csvs(data_dir: str):
    """
    Loads all CSV files from the specified data directory.
    Assumes each file is named '<station_id>.csv'.
    Returns a dictionary mapping station_id to its DataFrame.
    """
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    station_dfs = {}
    
    if not csv_files:
        print(f"WARNING: No CSV files found in {data_dir}. Please run load_data.py first.")
        
    for file in csv_files:
        basename = os.path.basename(file)
        station_id = basename.replace(".csv", "")
        
        df = pd.read_csv(file)
        
        # Ensure timestamp is datetime and set as index
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        elif 'Unnamed: 0' in df.columns:
            df.rename(columns={'Unnamed: 0': 'timestamp'}, inplace=True)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        elif df.index.name == 'times' or df.index.name == 'timestamp':
            df.index = pd.to_datetime(df.index)
        else:
            # Fallback assuming first column is time
            first_col = df.columns[0]
            df[first_col] = pd.to_datetime(df[first_col])
            df.set_index(first_col, inplace=True)
            
        station_dfs[station_id] = df
        
    return station_dfs
