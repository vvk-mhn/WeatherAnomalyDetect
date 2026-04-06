import torch
import torch.nn as nn
from src.models.autoencoder import create_model
from copy import deepcopy

def train_federated_autoencoders(datasets_per_station, A, station_ids, config):
    """
    Implements GTVMin-based Federated Training (alpha > 0).
    Uses the synchronous fixed-point iteration approach.
    """
    epochs = config['training']['epochs']
    lr = config['training']['lr']
    alpha = config['graph']['alpha']
    m_type = config['model']['type']
    latent_dim = config['model']['latent_dim']
    
    # Initialize models and optimizers
    models = {}
    optimizers = {}
    criterion = nn.MSELoss()
    
    for sid in station_ids:
        in_dim = datasets_per_station[sid]['input_dim']
        models[sid] = create_model(m_type, in_dim, latent_dim)
        optimizers[sid] = torch.optim.SGD(models[sid].parameters(), lr=lr)
        
    # Logging
    logs = {sid: {'train_loss': []} for sid in station_ids}
    
    for epoch in range(epochs):
        # 1. Snapshot all models at iteration t (Synchronous step)
        snapshot = {sid: deepcopy(models[sid].state_dict()) for sid in station_ids}
        
        epoch_losses = {sid: 0.0 for sid in station_ids}
        batches = {sid: 0 for sid in station_ids}
        
        # 2. Iterate through stations
        for i, sid in enumerate(station_ids):
            model = models[sid]
            optimizer = optimizers[sid]
            dataloader = datasets_per_station[sid]['train']
            
            for X_batch, _ in dataloader:
                optimizer.zero_grad()
                
                # Forward & local loss L_i
                recon = model(X_batch)
                loss = criterion(recon, X_batch)
                
                # Backprop to compute \nabla L_i
                loss.backward()
                
                # --- Fixed-Point Operator / Graph Coupling ---
                # w^(i, t+1) = w^(i,t) - lr * ( \nabla L_i + 2 * alpha * \sum A_ij (w_i - w_j) )
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            coupling = torch.zeros_like(param)
                            # Sum over neighbors
                            for j, neighbor_id in enumerate(station_ids):
                                weight = A[i, j]
                                if weight > 0:
                                    neighbor_param = snapshot[neighbor_id][name] # From snapshot!
                                    # (w_i - w_j)
                                    coupling += weight * (param.data - neighbor_param)
                                    
                            # Inject coupling penalty directly to gradient
                            param.grad.add_(2 * alpha * coupling)
                
                optimizer.step()
                
                epoch_losses[sid] += loss.item()
                batches[sid] += 1
                
        # Logging
        print(f"Epoch {epoch+1}/{epochs} | Federated GTVMin (alpha={alpha}) Loss: ", end="")
        avg_overall = 0
        for sid in station_ids:
            avg_loss = epoch_losses[sid] / max(1, batches[sid])
            logs[sid]['train_loss'].append(avg_loss)
            avg_overall += avg_loss
        print(f"{avg_overall / len(station_ids):.4f}")
            
    return models, logs
