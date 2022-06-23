# pylint: disable=missing-function-docstring, invalid-name, no-member

"""Content Multi Layer Perceptron used for users and services contents processing"""

import torch
from torch.nn import Module, Sequential, ReLU

from recommender.engines.base.base_neural_network import BaseNeuralNetwork


class ContentMLP(Module, BaseNeuralNetwork):
    """Content Multi Layer Perceptron used for users and services contents processing"""

    def __init__(self, layers_spec, user_emb_dim, service_emb_dim):
        super().__init__()

        input_dim = user_emb_dim + service_emb_dim
        output_dim = False

        layers = self._create_layers(
            input_dim, output_dim, layers_spec, inc_batchnorm=True, activation=ReLU
        )
        self.layers = Sequential(*layers)

    def forward(self, user_dense_tensors, service_dense_tensors):
        x = torch.cat([user_dense_tensors, service_dense_tensors], dim=1)

        output = self.layers(x)

        return output
