# pylint: disable=missing-module-docstring

import torch
from torch import nn


class HistoryEmbedder(torch.nn.Module):
    """
    Model used for transforming services history (list of service tensors
     in temporal order) into a history tensor.
     It should be used and trained inside both actor and critic.
    """

    def __init__(self, SE: int, num_layers: int = 3, dropout: float = 0.5):
        """
        NOTE: hidden_size is always equal to SE

        Args:
            SE: service embedding dimension
            num_layers: number of hidden layers
            dropout: a dropout probability
        """
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=SE,
            hidden_size=SE,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, services_history: torch.Tensor) -> torch.Tensor:
        """
        RNN is used for reducing history's N dimension.

        Args:
            services_history: history of services engaged by the
            user represented as a tensor of shape [N, SE]
            where N is the history length and SE is service content tensor embedding dim

        Returns:
            Embedded history of services engaged by the user as a tensor of shape [SE]
        """

        output, _ = self.rnn(services_history)
        return output[:, -1]
