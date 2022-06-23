# pylint: disable=missing-function-docstring, too-many-arguments, invalid-name, no-member

"""Multilayer Perceptron used in Neural Collaborative Filtering model"""

import torch
from torch.nn import Module, Embedding, Sequential, ReLU

from recommender.engines.base.base_neural_network import BaseNeuralNetwork


class MLP(Module, BaseNeuralNetwork):
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

        input_dim = user_ids_embedding_dim + service_ids_embedding_dim
        output_dim = False

        layers = self._create_layers(
            input_dim, output_dim, layers_spec, inc_batchnorm=True, activation=ReLU
        )
        self.layers = Sequential(*layers)

    def forward(self, users_ids, services_ids):
        mlp_user_tensor = self.mlp_user_embedder(users_ids)
        mlp_service_tensor = self.mlp_service_embedder(services_ids)

        x = torch.cat([mlp_user_tensor, mlp_service_tensor], dim=1)

        output = self.layers(x)

        return output
