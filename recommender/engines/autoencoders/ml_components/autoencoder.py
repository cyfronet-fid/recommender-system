"""Autoencoder class"""
from typing import Tuple, List
from itertools import chain

from torch.nn import Module, Sequential, Linear, BatchNorm1d, ReLU


USER_AE_MODEL = "User Autoencoder Model"
SERVICE_AE_MODEL = "Service Autoencoder Model"
ENCODER = "encoder"
DECODER = "decoder"


class AutoEncoder(Module):
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
            first_dim = features_dim if name == ENCODER else embedding_dim
            last_dim = embedding_dim if name == ENCODER else features_dim

            # The first layer
            layers = [Linear(first_dim, size[0]), BatchNorm1d(size[0]), ReLU()]
            self._get_middle_layers(layers, size)  # The middle layers
            layers += [Linear(size[-1], last_dim)]  # The last layer

            if name == ENCODER:
                self.encoder = Sequential(*layers)
            else:
                self.decoder = Sequential(*layers)

    def forward(self, features):
        """Forward"""
        embedding = self.encoder(features)
        reconstruction = self.decoder(embedding)

        return reconstruction

    @staticmethod
    def _get_middle_layers(layers: List, layers_sizes: Tuple[int]):
        """
        Get layers between the first and the last layer.

        Args:
            layers: a list which includes the first layer
            layers_sizes: a tuple of target layers sizes
        """
        layers += list(
            chain.from_iterable(
                [
                    [Linear(n_size, next_n_size), BatchNorm1d(next_n_size), ReLU()]
                    for n_size, next_n_size in zip(layers_sizes, layers_sizes[1:])
                ]
            )
        )
