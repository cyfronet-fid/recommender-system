"""Autoencoder class"""
from typing import Tuple

from torch.nn import Module, Sequential, ReLU

from recommender.engines.base.base_neural_network import BaseNeuralNetwork

USER_AE_MODEL = "User Autoencoder Model"
SERVICE_AE_MODEL = "Service Autoencoder Model"
ENCODER = "encoder"
DECODER = "decoder"


class AutoEncoder(Module, BaseNeuralNetwork):
    """An Autoencoder model"""

    def __init__(
        self,
        features_dim: int,
        embedding_dim: int,
        encoder_layer_sizes: Tuple[int, int] = (128, 64),
        decoder_layer_sizes: Tuple[int, int] = (64, 128),
    ):
        super().__init__()

        sizes = {ENCODER: encoder_layer_sizes, DECODER: decoder_layer_sizes}

        for name, size in sizes.items():
            input_dim = features_dim if name == ENCODER else embedding_dim
            output_dim = embedding_dim if name == ENCODER else features_dim

            layers = self._create_layers(
                input_dim, output_dim, size, inc_batchnorm=True, activation=ReLU
            )

            if name == ENCODER:
                self.encoder = Sequential(*layers)
            else:
                self.decoder = Sequential(*layers)

    def forward(self, features):
        """Forward"""
        embedding = self.encoder(features)
        reconstruction = self.decoder(embedding)

        return reconstruction
