# pylint: disable=missing-module-docstring, useless-super-delegation

import torch

from recommender.engine.rl_agent.models.lstm_model import LSTMModel

SEARCH_PHRASE_MODEL = "search_phrase_model"


class SearchPhraseModel(LSTMModel):
    """
    Model used for transforming search phrase subwords tensor [X, SPE]
        into a search phrase tensor [SPE]
        it should be used and trained inside both actor and critic.
    """

    def __init__(self, SPE: int, num_layers: int = 3, dropout: float = 0.5):
        """
        NOTE: hidden_size is always equal to SPE

        Args:
            SPE: search phrase embedding dimension
            num_layers: number of hidden layers
            dropout: a dropout probability
        """
        super().__init__(SPE, num_layers, dropout)

    def forward(self, temporal_data: torch.Tensor) -> torch.Tensor:
        """
        RNN is used for reducing search phrase's X dimension.

        Args:
            temporal_data: search phrase subwords tensor of shape [X, SPE]
            where X is the subwords length and SPE is search phrase embedding dim

        Returns:
            Embedded history of services engaged by the user as a tensor of shape [SE]
        """
        return super().forward(temporal_data)
