# pylint: disable=missing-module-docstring, useless-super-delegation, missing-class-docstring

import torch
from torch import nn


class LSTMModel(torch.nn.Module):
    def __init__(self, input_size: int, num_layers: int = 3, dropout: float = 0.5):
        """
        NOTE: hidden_size is always equal to SE

        Args:
            input_size: service embedding dimension
            num_layers: number of hidden layers
            dropout: a dropout probability
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=input_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, temporal_data: torch.Tensor) -> torch.Tensor:
        """
        Reduces temporal_data's temporal dimension.

        Args:
            temporal_data: history of services engaged by the
            user represented as a tensor of shape [N, input_size]
            where N is the length of the time steps

        Returns:
            Temporal data with reduced temporal dim engaged as
            a tensor of shape [input_size]
        """

        output, _ = self.lstm(temporal_data)
        return output[:, -1]
