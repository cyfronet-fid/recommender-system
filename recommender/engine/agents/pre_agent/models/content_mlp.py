# pylint: disable=missing-function-docstring, no-self-use, invalid-name, no-member

"""Content Multi Layer Perceptron used for users and services contents processing"""

import torch
from torch.nn import Module, Sequential, Linear, ReLU, BatchNorm1d


class ContentMLP(Module):
    """Content Multi Layer Perceptron used for users and services contents processing"""

    def __init__(self, user_embedder, service_embedder, layers_spec):
        super().__init__()

        self.user_embedder = user_embedder
        self.service_embedder = service_embedder

        user_emb_dim = user_embedder[-1].out_features
        service_emb_dim = service_embedder[-1].out_features

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

    def forward(self, users_contents, services_contents):
        users_contents_embeddings = self.user_embedder(users_contents)
        services_contents_embeddings = self.service_embedder(services_contents)

        x = torch.cat([users_contents_embeddings, services_contents_embeddings], dim=1)

        output = self.layers(x)

        return output
