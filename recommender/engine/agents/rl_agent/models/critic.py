# pylint: disable=invalid-name, too-many-arguments, no-member

"""Critic Model implementation"""
from itertools import chain
from typing import Tuple, Optional

import torch
from torch.nn import Linear, ReLU, Sequential

from recommender.errors import MissingComponentError, NoHistoryEmbedderForK
from recommender.engine.utils import load_last_module, NoSavedModuleError
from recommender.engine.agents.rl_agent.models.history_embedder import (
    HistoryEmbedder,
    MLP_HISTORY_EMBEDDER_V1,
    MLP_HISTORY_EMBEDDER_V2,
)


class Critic(torch.nn.Module):
    """Critic Model"""

    def __init__(
        self,
        K: int,
        SE: int,
        UE: int,
        I: int,
        history_embedder: Optional[HistoryEmbedder] = None,
        layer_sizes: Tuple[int] = (256, 512, 256),
    ):
        super().__init__()
        self.K = K
        self.SE = SE

        self.history_embedder = history_embedder

        self._load_models()

        layers = [Linear(UE + SE + I + K * SE, layer_sizes[0]), ReLU()]
        layers += list(
            chain.from_iterable(
                [
                    [Linear(n_size, next_n_size), ReLU()]  # BatchNorm1d(next_n_size)
                    for n_size, next_n_size in zip(layer_sizes, layer_sizes[1:])
                ]
            )
        )
        layers += [(Linear(layer_sizes[-1], 1))]

        self.network = Sequential(*layers)

    def forward(
        self, state: Tuple[(torch.Tensor,) * 3], action: torch.Tensor
    ) -> torch.Tensor:
        """
        Performs forward propagation.

        Args:
            state: Tuple of following tensors:
                user: Batch of embedded users content tensors of shape
                 [batch_size, UE].
                services_history: Batch of services history tensors of shape
                 [batch_size, N, SE].
                search_data_mask: Batch of search data masks of shape
                [batch_size, I]
            action: Batch of encoded actor weight tensors of shape
             [batch_size, K, SE].
                where:
                  - UE is user content tensor embedding dim
                  - N is user clicked services history length
                  - SE is service content tensor embedding dim
                  - K is the number of services in the recommendation
                  - I is the itemspace (services) size

        Returns:
            action_value: Batch of action values - tensor of shape [batch_size, 1]
        """
        user, services_history, mask = state

        services_history = self.history_embedder(services_history)
        action = action.reshape(action.shape[0], -1)

        tensors_to_concat = [user, services_history, mask, action]
        network_input = torch.cat(tensors_to_concat, dim=1)
        action_value = self.network(network_input)

        return action_value

    def _load_models(self):
        try:
            if self.K == 3:
                history_embedder_name = MLP_HISTORY_EMBEDDER_V1
            elif self.K == 2:
                history_embedder_name = MLP_HISTORY_EMBEDDER_V2
            else:
                raise NoHistoryEmbedderForK
            self.history_embedder = self.history_embedder or load_last_module(
                history_embedder_name
            )

        except NoSavedModuleError as no_saved_module:
            raise MissingComponentError from no_saved_module
