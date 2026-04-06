import yaml
import torch
import numpy as np
import random
import os

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def setup_results_dir(config):
    d = config['run']['results_dir']
    if not os.path.exists(d):
        os.makedirs(d)
        
def save_logs(logs, results, path):
    # Dummy saver
    import json
    with open(path, 'w') as f:
        json.dump({'logs': logs, 'results': results}, f, indent=4)
