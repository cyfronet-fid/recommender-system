# pylint: disable=invalid-name, too-many-arguments, no-member

"""Critic Model implementation"""
from itertools import chain
from typing import Tuple, Optional

import torch
from torch.nn import Linear, ReLU, Sequential

from recommender.errors import (
    MissingComponentError,
    NoHistoryEmbedderForK,
    NoSearchPhraseEmbedderForK,
)
from recommender.engine.utils import load_last_module, NoSavedModuleError
from recommender.engine.agents.rl_agent.models.action_embedder import ActionEmbedder
from recommender.engine.agents.rl_agent.models.search_phrase_embedder import (
    SearchPhraseEmbedder,
    SEARCH_PHRASE_EMBEDDER_V1,
    SEARCH_PHRASE_EMBEDDER_V2,
)
from recommender.engine.agents.rl_agent.models.history_embedder import (
    HistoryEmbedder,
    HISTORY_EMBEDDER_V1,
    HISTORY_EMBEDDER_V2,
)


class Critic(torch.nn.Module):
    """Critic Model"""

    def __init__(
        self,
        K: int,
        SE: int,
        UE: int,
        SPE: int,
        search_phrase_embedder: Optional[SearchPhraseEmbedder] = None,
        history_embedder: Optional[HistoryEmbedder] = None,
        action_embedder: Optional[ActionEmbedder] = None,
        layer_sizes: Tuple[int] = (256, 512, 256),
    ):
        super().__init__()
        self.K = K
        self.SE = SE

        self.search_phrase_embedder = search_phrase_embedder
        self.history_embedder = history_embedder
        self.action_embedder = action_embedder

        self._load_models()

        layers = [
            Linear(3 * SE + UE + SPE, layer_sizes[0]),
            ReLU(),
        ]  # 3 * SE because it includes filters, services and action
        layers += list(
            chain.from_iterable(
                [
                    [Linear(n_size, next_n_size), ReLU()]
                    for n_size, next_n_size in zip(layer_sizes, layer_sizes[1:])
                ]
            )
        )
        layers += [(Linear(layer_sizes[-1], 1))]

        self.network = Sequential(*layers)

    def forward(self, state: Tuple[torch.Tensor], action: torch.Tensor) -> torch.Tensor:
        """
        Performs forward propagation.

        Args:
            state: Tuple of following tensors:
                user: Batch of embedded users content tensors of shape
                 [batch_size, UE].
                services_history: Batch of services history tensors of shape
                 [batch_size, N, SE].
                filters: Batch of embedded filters tensors of shape
                 [batch_size, SE].
                search_phrase: Batch of encoded search phrase tensors of shape
                 [batch_size, SPE].
            action: Batch of encoded actions tensors of shape
             [batch_size, K, SE].
                where:
                  - UE is user content tensor embedding dim
                  - N is user clicked services history length
                  - SE is service content tensor embedding dim
                  - SPE is search phrase tensor embedding dim
                  - K is the nmber of services in the recommendation

        Returns:
            action_value: Batch of action values - tensor of shape [batch_size, 1]
        """
        user, services_history, filters, search_phrase = state

        services_history = self.history_embedder(services_history)
        search_phrase = self.search_phrase_embedder(search_phrase)
        action = self.action_embedder(action)

        tensors_to_concat = [user, services_history, filters, search_phrase, action]
        network_input = torch.cat(tensors_to_concat, dim=1)
        action_value = self.network(network_input)

        return action_value

    def _load_models(self):
        try:
            if self.K == 3:
                history_embedder_name = HISTORY_EMBEDDER_V1
            elif self.K == 2:
                history_embedder_name = HISTORY_EMBEDDER_V2
            else:
                raise NoHistoryEmbedderForK
            self.history_embedder = self.history_embedder or load_last_module(
                history_embedder_name
            )

            if self.K == 3:
                search_phrase_embedder_name = SEARCH_PHRASE_EMBEDDER_V1
            elif self.K == 2:
                search_phrase_embedder_name = SEARCH_PHRASE_EMBEDDER_V2
            else:
                raise NoSearchPhraseEmbedderForK
            self.search_phrase_embedder = (
                self.search_phrase_embedder
                or load_last_module(search_phrase_embedder_name)
            )

            self.action_embedder = self.action_embedder or ActionEmbedder(self.SE)
        except NoSavedModuleError as no_saved_module:
            raise MissingComponentError from no_saved_module
