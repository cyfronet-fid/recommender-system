"""Autoencoder class"""
from torch.nn import Module, Sequential, Linear, BatchNorm1d, ReLU


USER_AE_MODEL = "User Autoencoder Model"
SERVICE_AE_MODEL = "Service Autoencoder Model"


class AutoEncoder(Module):
    """An Autoencoder model"""

    def __init__(self, features_dim, embedding_dim):
        super().__init__()

        self.encoder = Sequential(
            Linear(features_dim, 128),
            BatchNorm1d(128),
            ReLU(),
            Linear(128, 64),
            BatchNorm1d(64),
            ReLU(),
            Linear(64, embedding_dim),
        )

        self.decoder = Sequential(
            Linear(embedding_dim, 64),
            BatchNorm1d(64),
            ReLU(),
            Linear(64, 128),
            BatchNorm1d(128),
            ReLU(),
            Linear(128, features_dim),
        )

    def forward(self, features):
        """Forward"""
        embedding = self.encoder(features)
        reconstruction = self.decoder(embedding)

        return reconstruction
