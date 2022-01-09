# pylint: disable=missing-module-docstring, invalid-name, too-many-arguments, no-member

from typing import Tuple

import torch
from torch.nn import Module, Sequential, ReLU

from recommender.engines.persistent_mixin import Persistent
from recommender.engines.rl.ml_components.history_embedder import HistoryEmbedder
from recommender.engines.base.base_neural_network import BaseNeuralNetwork

ACTOR_V1 = "actor_v1"
ACTOR_V2 = "actor_v2"


class Actor(Module, Persistent, BaseNeuralNetwork):
    """Actor neural network representing a deterministic policy used by RLAgent"""

    def __init__(
        self,
        K: int,
        SE: int,
        UE: int,
        I: int,
        history_embedder: HistoryEmbedder,
        layer_sizes: Tuple[int] = (256, 512, 256),
        act_max: float = 1.0,
        act_min: float = -1.0,
    ):
        """
        Args:
            K: number of services to recommend
            SE: service embedding dimension
            UE: user embedding dimension
            I: itemspace size
            history_embedder: pytorch module implementing history embedding
            layer_sizes: list containing number of neurons in each hidden layer
        """
        super().__init__()

        self.K = K
        self.SE = SE

        self.history_embedder = history_embedder

        input_dim = UE + SE + I
        output_dim = K * SE

        layers = self._create_layers(
            input_dim, output_dim, layer_sizes, inc_batchnorm=True, activation=ReLU
        )

        self.network = Sequential(*layers)
        self.act_max = act_max
        self.act_min = act_min

    def forward(self, state: Tuple[(torch.Tensor,) * 3]) -> torch.Tensor:
        """
        Performs forward propagation.

        Args:
            state:
                user: Embedded user content tensor of shape
                 [batch_size, UE]
                services_history: Services history tensor of shape
                 [batch_size, N, SE]
                search_data_mask: Batch of search data masks of shape
                 [batch_size, I]

        Returns:
            weights: Weights tensor used for choosing action from the
             itemspace.
        """

        user, services_history, mask = state

        services_history = self.history_embedder(services_history)
        x = torch.cat([user, services_history, mask], dim=1)
        x = self.network(x)

        weights = x.reshape(-1, self.K, self.SE)
        weights = torch.tanh(weights)

        # https://stackoverflow.com/questions/345187/math-mapping-numbers
        # Mapping to range (self.act_min, self.act_max) using 'two-point form'
        # linear transform.
        # Old bounds are (-1, 1) since weights come from a hyperbolic tangent
        scaled_weights = (
            (weights + 1) * (self.act_max - self.act_min)
        ) / 2 + self.act_min

        return scaled_weights
