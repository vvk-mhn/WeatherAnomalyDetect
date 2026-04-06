import torch
import torch.nn as nn

class SmallAutoencoder(nn.Module):
    """
    A lightweight structural variant of the Autoencoder.
    """
    def __init__(self, input_dim, latent_dim=16):
        super(SmallAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.decoder(self.encoder(x))


class LargeAutoencoder(nn.Module):
    """
    A deeper structural variant of the Autoencoder.
    """
    def __init__(self, input_dim, latent_dim=16):
        super(LargeAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.decoder(self.encoder(x))

        
def create_model(model_type, input_dim, latent_dim):
    if model_type == 'small':
        return SmallAutoencoder(input_dim, latent_dim)
    elif model_type == 'large':
        return LargeAutoencoder(input_dim, latent_dim)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
