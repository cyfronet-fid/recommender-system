# pylint: disable=missing-function-docstring, too-many-arguments, invalid-name, no-member, no-self-use

"""Multilayer Perceptron used in Neural Collaborative Filtering model"""

import torch
from torch.nn import Module, Embedding, Linear, ReLU, BatchNorm1d, Sequential


class MLP(Module):
    """Multilayer Perceptron"""

    def __init__(
        self,
        users_max_id,
        services_max_id,
        user_ids_embedding_dim,
        service_ids_embedding_dim,
        layers_spec,
    ):
        super().__init__()

        self.mlp_user_embedder = Embedding(users_max_id + 1, user_ids_embedding_dim)
        self.mlp_service_embedder = Embedding(
            services_max_id + 1, service_ids_embedding_dim
        )

        self.layers = self.create_layers(
            user_ids_embedding_dim, service_ids_embedding_dim, layers_spec
        )

    def create_layers(
        self, user_ids_embedding_dim, service_ids_embedding_dim, layers_spec
    ):
        layers = []
        first = torch.nn.Sequential(
            Linear(user_ids_embedding_dim + service_ids_embedding_dim, layers_spec[0]),
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

    def forward(self, users_ids, services_ids):
        mlp_user_tensor = self.mlp_user_embedder(users_ids)
        mlp_service_tensor = self.mlp_service_embedder(services_ids)

        x = torch.cat([mlp_user_tensor, mlp_service_tensor], dim=1)

        output = self.layers(x)

        return output
