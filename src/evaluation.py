import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

def evaluate_models(models, datasets_per_station):
    """
    Evaluates models on their respective test sets.
    Returns dictionary with AUCs and anomaly detection metrics per station.
    """
    results = {}
    criterion = nn.MSELoss(reduction='none') # per-sample loss
    
    for sid, model in models.items():
        model.eval()
        loader = datasets_per_station[sid]['test']
        
        all_scores = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in loader:
                recon = model(X_batch)
                # MSE per sample over features
                loss = criterion(recon, X_batch).mean(dim=1).numpy()
                
                all_scores.extend(loss)
                all_labels.extend(y_batch.numpy())
                
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        
        # Calculate AUC if there are both positive and negative samples
        if len(np.unique(all_labels)) > 1:
            roc_auc = roc_auc_score(all_labels, all_scores)
            pr_auc = average_precision_score(all_labels, all_scores)
        else:
            roc_auc = float('nan')
            pr_auc = float('nan')
            
        results[sid] = {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'test_loss': np.mean(all_scores),
            'num_anomalies': int(np.sum(all_labels))
        }
        
    return results
