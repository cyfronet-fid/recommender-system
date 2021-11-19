# pylint: disable=missing-module-docstring, invalid-name, too-many-arguments, no-member

from typing import Tuple
from itertools import chain

import torch
from torch import nn
from torch.nn import BatchNorm1d

from recommender.engines.persistent_mixin import Persistent
from recommender.engines.rl.ml_components.history_embedder import HistoryEmbedder

ACTOR_V1 = "actor_v1"
ACTOR_V2 = "actor_v2"


class Actor(nn.Module, Persistent):
    """Actor neural network representing a deterministic policy used by RLAgent"""

    def __init__(
        self,
        K: int,
        SE: int,
        UE: int,
        I: int,
        history_embedder: HistoryEmbedder,
        layer_sizes: Tuple[int] = (256, 512, 256),
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

        layers = [
            nn.Linear(UE + SE + I, layer_sizes[0]),
            nn.ReLU(),
            BatchNorm1d(layer_sizes[0]),
        ]
        layers += list(
            chain.from_iterable(
                [
                    [
                        nn.Linear(n_size, next_n_size),
                        nn.ReLU(),
                        BatchNorm1d(next_n_size),
                    ]
                    for n_size, next_n_size in zip(layer_sizes, layer_sizes[1:])
                ]
            )
        )
        layers += [(nn.Linear(layer_sizes[-1], K * SE))]

        self.network = nn.Sequential(*layers)

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
        return weights