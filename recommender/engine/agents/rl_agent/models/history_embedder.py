# pylint: disable=missing-module-docstring, useless-super-delegation
from itertools import chain

import torch
from torch import nn
from torch.nn import BatchNorm1d

from recommender.engine.agents.rl_agent.models.base_lstm_embedder import (
    BaseLSTMEmbedder,
)

import torch.nn.functional as F

HISTORY_EMBEDDER_V1 = "history embedder v1"
HISTORY_EMBEDDER_V2 = "history embedder v2"


# class HistoryEmbedder(BaseLSTMEmbedder):
#     """
#     Model used for transforming services history tensor [N, SE]
#     into a history tensor [SE].
#      It should be used and trained inside both actor and critic.
#     """
#
#     def __init__(self, SE: int, num_layers: int = 3, dropout: float = 0.5):
#         """
#         NOTE: hidden_size is always equal to SE
#
#         Args:
#             SE: service embedding dimension
#             num_layers: number of hidden layers
#             dropout: a dropout probability
#         """
#         super().__init__(SE, num_layers, dropout)
#
#     def forward(self, temporal_data: torch.Tensor) -> torch.Tensor:
#         """
#         RNN is used for reducing history's N dimension.
#
#         Args:
#             temporal_data: history of services engaged by the
#             user represented as a tensor of shape [N, SE]
#             where N is the history length and SE is service content tensor embedding dim
#
#         Returns:
#             Embedded history of services engaged by the user as a tensor of shape [SE]
#         """
#
#         return super().forward(temporal_data)

class HistoryEmbedder(torch.nn.Module):
    """
    Model used for transforming services history tensor [N, SE]
    into a history tensor [SE].
     It should be used and trained inside both actor and critic.
    """

    def __init__(self, SE: int, N=20, layer_sizes=(256, 128), num_layers: int = 3, dropout: float = 0.5):
        """
        NOTE: hidden_size is always equal to SE

        Args:
            SE: service embedding dimension
            num_layers: number of hidden layers
            dropout: a dropout probability
        """
        super().__init__()
        self.N = N # TODO: make it even wiser
        self.SE = SE
        # self.network = torch.nn.Linear(self.N * SE, SE)

        layers = [torch.nn.Linear(self.N * SE, layer_sizes[0]), nn.ReLU()] # , BatchNorm1d(layer_sizes[0])
        layers += list(
            chain.from_iterable(
                [
                    [nn.Linear(n_size, next_n_size), nn.ReLU()] # , BatchNorm1d(next_n_size)
                    for n_size, next_n_size in zip(layer_sizes, layer_sizes[1:])
                ]
            )
        )
        layers += [(nn.Linear(layer_sizes[-1], SE))]

        self.network = nn.Sequential(*layers)

    def forward(self, temporal_data: torch.Tensor) -> torch.Tensor:
        """
        RNN is used for reducing history's N dimension.

        Args:
            temporal_data: history of services engaged by the
            user represented as a tensor of shape [N, SE]
            where N is the history length and SE is service content tensor embedding dim

        Returns:
            Embedded history of services engaged by the user as a tensor of shape [SE]
        """

        B = temporal_data.shape[0]

        concated_history = temporal_data.reshape((B, -1))
        zeros = torch.zeros((B, self.N*self.SE))
        zeros[:, :concated_history.shape[1]] = concated_history
        padded_history = zeros.to(temporal_data.device)

        x = self.network(padded_history)
        x = F.relu(x)

        return x
