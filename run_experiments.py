import os
import copy
from src.utils import load_config, set_seed, setup_results_dir, save_logs
from src.data_interface import load_fmi_csvs
from src.preprocessing import build_station_datasets
from src.graph import build_graph
from src.local_baseline import train_local_autoencoders
from src.fl_training import train_federated_autoencoders
from src.evaluation import evaluate_models
import json

def run_experiment():
    print("--- 1. Initialization ---")
    config = load_config('config/default_config.yaml')
    set_seed(config['run']['seed'])
    setup_results_dir(config)
    
    print("--- 2. Loading Data ---")
    station_dfs = load_fmi_csvs(config['run']['data_dir'])
    print(f"Loaded {len(station_dfs)} stations.")
    
    print("--- 3. Preprocessing ---")
    datasets = build_station_datasets(station_dfs, config)
    
    print("--- 4. Graph Construction ---")
    A, station_ids = build_graph(station_dfs, config)
    print("Graph built successfully.")
    
    # -----------------------------------------------------
    # System A: Local Baseline (alpha = 0)
    # -----------------------------------------------------
    print("\n========== SYSTEM A: LOCAL BASELINE (alpha=0) ==========")
    models_local, logs_local = train_local_autoencoders(datasets, station_ids, config)
    res_local = evaluate_models(models_local, datasets)
    
    # -----------------------------------------------------
    # System B: Federated GTVMin (alpha > 0)
    # -----------------------------------------------------
    config_b = copy.deepcopy(config)
    # the config dict already has config['graph']['alpha'] > 0
    print(f"\n========== SYSTEM B: FEDERATED GTVMin (alpha={config_b['graph']['alpha']}) ==========")
    models_fed, logs_fed = train_federated_autoencoders(datasets, A, station_ids, config_b)
    res_fed = evaluate_models(models_fed, datasets)
    
    # -----------------------------------------------------
    # System C: Federated GTVMin (Large Model Variant)
    # -----------------------------------------------------
    config_c = copy.deepcopy(config)
    config_c['model']['type'] = 'large' # structural design choice requirement
    print(f"\n========== SYSTEM C: FEDERATED GTVMin LARGE (alpha={config_c['graph']['alpha']}) ==========")
    models_fed_l, logs_fed_l = train_federated_autoencoders(datasets, A, station_ids, config_c)
    res_fed_l = evaluate_models(models_fed_l, datasets)
    
    # Save Results
    print("\n========== SAVING RESULTS ==========")
    save_logs(logs_local, res_local, os.path.join(config['run']['results_dir'], "results_local.json"))
    save_logs(logs_fed, res_fed, os.path.join(config['run']['results_dir'], "results_fed.json"))
    save_logs(logs_fed_l, res_fed_l, os.path.join(config['run']['results_dir'], "results_fed_large.json"))
    print(f"Results saved in {config['run']['results_dir']}/")

if __name__ == "__main__":
    run_experiment()
