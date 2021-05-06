# pylint: disable=missing-module-docstring, useless-super-delegation

import torch

from recommender.engine.rl_agent.models.lstm_model import LSTMModel

HISTORY_MODEL = "history_model"


class HistoryModel(LSTMModel):
    """
    Model used for transforming services history tensor [N, SE]
    into a history tensor [SE].
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
        super().__init__(SE, num_layers, dropout)

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
        return super().forward(temporal_data)
