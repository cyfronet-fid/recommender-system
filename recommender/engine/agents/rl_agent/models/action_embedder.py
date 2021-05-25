# pylint: disable=missing-module-docstring, useless-super-delegation, arguments-differ

import torch

from recommender.engine.agents.rl_agent.models.base_lstm_embedder import (
    BaseLSTMEmbedder,
)

ACTION_EMBEDDER = "Action Embedder"


class ActionEmbedder(BaseLSTMEmbedder):
    """
    Model used for transforming action tensor [K, SE]
    into a embedded action tensor [SE].
     It should be used and trained inside critic.
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

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        """
        RNN is used for reducing action's K dimension.

        Args:
            action: action tensor of shape [K, SE]
            where K is the number of services in recommendation

        Returns:
            Embedded action of shape [SE]
        """
        return super().forward(action)
