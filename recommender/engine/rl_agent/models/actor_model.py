# pylint: disable=missing-module-docstring, invalid-name, too-many-arguments, no-member

from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F

from recommender.engine.rl_agent.models.history_embedder import HistoryEmbedder


class ActorModel(nn.Module):
    """Actor neural network representing a deterministic policy used by RLAgent"""

    def __init__(
        self,
        K: int,
        SE: int,
        UE: int,
        FE: int,
        SPE: int,
        history_embedder: HistoryEmbedder,
        layer_sizes: Tuple[int] = (256, 512, 256),
    ):
        """
        Args:
            K: number of services to recommend
            SE: service embedding dimension
            UE: user embedding dimension
            FE: filter embedding dimension
            SPE search phrase embedding dimension
            history_embedder: pytorch module implementing history embedding
            layer_sizes: list containing number of neurons in each hidden layer
        """
        super().__init__()
        self.K = K
        self.SE = SE
        self.history_embedder = history_embedder

        self.layers = [nn.Linear(SE + UE + FE + SPE, layer_sizes[0])]
        self.layers += [
            nn.Linear(n_size, next_n_size)
            for n_size, next_n_size in zip(layer_sizes, layer_sizes[1:])
        ]
        self.layers += [(nn.Linear(layer_sizes[-1], K * SE))]

    def forward(
        self,
        user: torch.Tensor,
        services_history: torch.Tensor,
        filters: torch.Tensor,
        search_phrase: torch.Tensor,
    ) -> torch.Tensor:
        """
        Performs forward propagation.

        Args:
            user: Embedded user content tensor of shape [UE]
            services_history: Services history tensor of shape [N, SE]
            filters: Embedded filters tensor of shape [FE]
            search_phrase: Embedded search phrase tensor of shape [SPE]

        """

        embedded_services_history = self.history_embedder(services_history)
        x = torch.cat([embedded_services_history, user, filters, search_phrase], dim=1)

        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)

        weights = x.reshape(-1, self.K, self.SE)
        return weights
