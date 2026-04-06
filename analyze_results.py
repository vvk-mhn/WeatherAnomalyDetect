import json
import matplotlib.pyplot as plt
import numpy as np

def analyze():
    try:
        with open("results/results_local.json") as f: res_loc = json.load(f)["results"]
        with open("results/results_fed.json") as f: res_fed = json.load(f)["results"]
        with open("results/results_fed_large.json") as f: res_large = json.load(f)["results"]
    except Exception as e:
        print("Run run_experiments.py first computationally.")
        return

    stations = list(res_loc.keys())
    
    auc_loc = []
    auc_fed = []
    auc_large = []
    
    for s in stations:
        if not np.isnan(res_loc[s]['roc_auc']):
            auc_loc.append(res_loc[s]['roc_auc'])
            auc_fed.append(res_fed[s]['roc_auc'])
            auc_large.append(res_large[s]['roc_auc'])
            
    print("====== AVERAGE ROC-AUC ACROSS VALID STATIONS ======")
    print(f"System A (Local)      : {np.mean(auc_loc):.4f}")
    print(f"System B (Fed Small)  : {np.mean(auc_fed):.4f}")
    print(f"System C (Fed Large)  : {np.mean(auc_large):.4f}")
    
    # Plotting comparison
    fig, ax = plt.subplots(figsize=(8,6))
    x = np.arange(len(auc_loc))
    width = 0.25
    
    ax.bar(x - width, auc_loc, width, label='Local')
    ax.bar(x, auc_fed, width, label='Fed Small')
    ax.bar(x + width, auc_large, width, label='Fed Large')
    
    ax.set_ylabel('ROC-AUC')
    ax.set_title('Anomaly Detection Performance by Station')
    ax.set_xticks(x)
    ax.set_xticklabels([f"S-{i}" for i in range(len(auc_loc))])
    ax.legend()
    
    plt.savefig('results/auc_comparison.png')
    print("Saved plot to results/auc_comparison.png")

if __name__ == "__main__":
    analyze()
