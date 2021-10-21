# pylint: disable=invalid-name

"""This module defines the history embedder classes"""
from itertools import chain
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch import nn


class HistoryEmbedder(torch.nn.Module, ABC):
    """History Embedder base class"""

    @abstractmethod
    def forward(self, service_history: torch.Tensor) -> torch.Tensor:
        """Reduces the temporal dimension of the service history"""


class LSTMHistoryEmbedder(HistoryEmbedder):
    """
    Model used for transforming services history tensor [N, SE]
    into a history tensor [SE] using an LSTM network.
    It should be used and trained inside both actor and critic.
    """

    def __init__(self, SE: int, num_layers: int = 3, dropout: float = 0.2):

        """
        NOTE: Hidden size is always equal to SE, to ensure proper output dimension.

        Args:
            SE: service embedding dimension
            num_layers: number of hidden layers
            dropout: a dropout probability
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=SE,
            hidden_size=SE,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, service_history: torch.Tensor) -> torch.Tensor:
        """
        Reduces service_history's temporal dimension using LSTM network.

        Args:
            service_history: history of services engaged by the
            user represented as a tensor of shape [N, input_size]
            where N is the length of the time steps

        Returns:
            service_history with reduced temporal dim of shape [input_size]
        """

        output, _ = self.lstm(service_history)
        return output[:, -1]


class MLPHistoryEmbedder(HistoryEmbedder):
    """
    Model used for transforming services history tensor [N, SE]
    into a history tensor [SE], using a simple feed forward network.
    It should be used and trained inside both actor and critic.
    """

    def __init__(self, SE: int, N=20, layer_sizes=(256, 128)):
        """
        Args:
            SE: service embedding dimension
            N: upper bound on history length
            layer_sizes: list of layers to use in a network
        """
        super().__init__()
        self.N = N
        self.SE = SE

        layers = [torch.nn.Linear(self.N * SE, layer_sizes[0]), nn.ReLU()]
        layers += list(
            chain.from_iterable(
                [
                    [nn.Linear(n_size, next_n_size), nn.ReLU()]
                    for n_size, next_n_size in zip(layer_sizes, layer_sizes[1:])
                ]
            )
        )
        layers += [(nn.Linear(layer_sizes[-1], SE))]

        self.network = nn.Sequential(*layers)

    def forward(self, service_history: torch.Tensor) -> torch.Tensor:
        """
        Reduces service_history's temporal dimension using MLP feed forward network.

        Args:
            service_history: history of services engaged by the
            user represented as a tensor of shape [N, SE]
            where N is the history length and SE is service content
             tensor embedding dim

        Returns:
            Processed history of services engaged by the user as a tensor of shape [SE]
        """

        B = service_history.shape[0]

        merged_history = service_history.reshape((B, -1))
        zeros = torch.zeros((B, self.N * self.SE))
        zeros[:, : merged_history.shape[1]] = merged_history
        padded_history = zeros.to(service_history.device)

        x = self.network(padded_history)
        x = F.relu(x)

        return x
