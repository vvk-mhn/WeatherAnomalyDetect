import torch
import torch.nn as nn
from src.models.autoencoder import create_model

def train_local_autoencoders(datasets_per_station, station_ids, config):
    """
    Implements the Baseline System A: independent local models (alpha = 0).
    No coupling term is added.
    """
    epochs = config['training']['epochs']
    lr = config['training']['lr']
    m_type = config['model']['type']
    latent_dim = config['model']['latent_dim']
    
    models = {}
    criterion = nn.MSELoss()
    logs = {sid: {'train_loss': []} for sid in station_ids}
    
    for sid in station_ids:
        in_dim = datasets_per_station[sid]['input_dim']
        model = create_model(m_type, in_dim, latent_dim)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            batches = 0
            
            for X_batch, _ in datasets_per_station[sid]['train']:
                optimizer.zero_grad()
                recon = model(X_batch)
                loss = criterion(recon, X_batch)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                batches += 1
                
            logs[sid]['train_loss'].append(epoch_loss / max(1, batches))
            
        models[sid] = model
        print(f"Station {sid} - Local Training Complete. Final Loss: {logs[sid]['train_loss'][-1]:.4f}")
        
    return models, logs
