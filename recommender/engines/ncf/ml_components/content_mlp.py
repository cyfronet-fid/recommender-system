# pylint: disable=missing-function-docstring, no-self-use, invalid-name, no-member

"""Content Multi Layer Perceptron used for users and services contents processing"""

import torch
from torch.nn import Module, Sequential, Linear, ReLU, BatchNorm1d


class ContentMLP(Module):
    """Content Multi Layer Perceptron used for users and services contents processing"""

    def __init__(self, layers_spec, user_emb_dim, service_emb_dim):
        super().__init__()

        self.layers = self.create_layers(user_emb_dim, service_emb_dim, layers_spec)

    def create_layers(self, user_emb_dim, service_emb_dim, layers_spec):
        layers = []
        first = Sequential(
            Linear(user_emb_dim + service_emb_dim, layers_spec[0]),
            ReLU(),
            BatchNorm1d(layers_spec[0]),
        )
        layers.append(first)

        for i in range(len(layers_spec) - 1):
            current = Sequential(
                Linear(layers_spec[i], layers_spec[i + 1]),
                ReLU(),
                BatchNorm1d(layers_spec[i + 1]),
            )
            layers.append(current)

        layers = Sequential(*layers)

        return layers

    def forward(self, user_dense_tensors, service_dense_tensors):
        x = torch.cat([user_dense_tensors, service_dense_tensors], dim=1)

        output = self.layers(x)

        return output
