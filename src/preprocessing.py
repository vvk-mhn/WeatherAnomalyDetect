import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class WindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def label_anomalies(df_window, config):
    """
    Generates rule-based binary anomaly labels for a given window sequence.
    Label 1 = Anomaly, 0 = Normal.
    """
    w_thresh = config['data']['anomaly_thresh'].get('wind_speed', 15.0)
    t_thresh = config['data']['anomaly_thresh'].get('temp_drop', 10.0)
    
    is_anomaly = 0
    # Rule 1: Storm anomaly
    if 'Wind speed' in df_window.columns:
        if df_window['Wind speed'].max() >= w_thresh:
            is_anomaly = 1
            
    # Rule 2: Frost/Extreme Temp Drop
    if 'Air temperature' in df_window.columns:
        temp_max = df_window['Air temperature'].max()
        temp_min = df_window['Air temperature'].min()
        if (temp_max - temp_min) >= t_thresh:
            is_anomaly = 1
            
    return is_anomaly

def build_station_datasets(station_dfs, config):
    """
    Builds sliding window datasets for each station.
    Returns:
      datasets_per_station: dict {station_id: {'train': DataLoader, 'val': DataLoader, 'test': DataLoader, 'input_dim': int}}
    """
    features = config['data']['features']
    window_size = config['data']['window_size']
    batch_size = config['data']['batch_size']
    
    # Global Min/Max for uniform normalization across nodes
    all_data = pd.concat([df[features] for df in station_dfs.values()])
    f_min = all_data.min()
    f_max = all_data.max()
    
    datasets_per_station = {}
    
    for sid, df in station_dfs.items():
        if len(df) <= window_size:
            continue
            
        # Normalize Data
        df_norm = (df[features] - f_min) / (f_max - f_min + 1e-8)
        
        arr = df_norm.values
        X_all, Y_all = [], []
        
        # Sliding window
        for i in range(len(arr) - window_size):
            X_all.append(arr[i : i + window_size].flatten())
            # Use raw unnormalized data for labeling
            raw_win = df[features].iloc[i : i + window_size]
            y_val = label_anomalies(raw_win, config)
            Y_all.append(y_val)
            
        X_all = np.array(X_all)
        Y_all = np.array(Y_all)
        
        # Split (Temporal)
        n = len(X_all)
        test_idx = int(n * (1 - config['data']['test_split']))
        val_idx = int(test_idx * (1 - config['data']['val_split']))
        
        X_train, y_train = X_all[:val_idx], Y_all[:val_idx]
        X_val, y_val = X_all[val_idx:test_idx], Y_all[val_idx:test_idx]
        X_test, y_test = X_all[test_idx:], Y_all[test_idx:]
        
        datasets_per_station[sid] = {
            'train': DataLoader(WindowDataset(X_train, y_train), batch_size=batch_size, shuffle=True),
            'val': DataLoader(WindowDataset(X_val, y_val), batch_size=batch_size, shuffle=False),
            'test': DataLoader(WindowDataset(X_test, y_test), batch_size=batch_size, shuffle=False),
            'input_dim': window_size * len(features)
        }
    
    return datasets_per_station
