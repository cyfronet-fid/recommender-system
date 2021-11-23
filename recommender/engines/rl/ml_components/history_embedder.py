# pylint: disable=invalid-name

"""This module defines the history embedder classes"""
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import ReLU

from recommender.engines.base.base_neural_network import BaseNeuralNetwork


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


class MLPHistoryEmbedder(HistoryEmbedder, BaseNeuralNetwork):
    """
    Model used for transforming services history tensor [N, SE]
    into a history tensor [SE], using a simple feed forward network.
    It should be used and trained inside both actor and critic.
    """

    def __init__(self, SE: int, max_N: int, layer_sizes=(256, 128)):
        """
        Args:
            SE: service embedding dimension
            max_N: upper bound on history length
            layer_sizes: list of layers to use in a network
        """
        super().__init__()
        self.max_N = max_N
        self.SE = SE

        input_dim = self.max_N * SE
        output_dim = SE

        layers = self._create_layers(
            input_dim, output_dim, layer_sizes, inc_batchnorm=False, activation=ReLU
        )

        self.network = nn.Sequential(*layers)

    def _align_history(self, service_history):
        B, current_N, SE = service_history.shape
        missing_N = self.max_N - current_N

        if missing_N > 0:
            padding = torch.zeros(B, missing_N, SE)
            aligned_history = torch.cat((service_history, padding), dim=1)
        else:
            aligned_history = service_history[:, -self.max_N :, :]

        return aligned_history

    def forward(self, service_history: torch.Tensor) -> torch.Tensor:
        """
        Reduces service_history's temporal dimension using MLP feed forward network.

        Args:
            service_history: history of services engaged by the
            user represented as a tensor of shape [B, N, SE]
            where N is the history length (not constant) and SE is service content
             tensor embedding dim

        Returns:
            Processed history of services engaged by the user
                as a tensor of shape [B, SE]
        """

        aligned_history = self._align_history(service_history)
        aligned_history = aligned_history.reshape(aligned_history.shape[0], -1)

        x = self.network(aligned_history)
        x = F.relu(x)

        return x
