# pylint: disable=missing-module-docstring, invalid-name, too-many-arguments, no-member

from typing import Tuple, Optional
from itertools import chain

import torch
from torch import nn

from recommender.engine.utils import load_last_module, NoSavedModuleError
from recommender.errors import (
    MissingComponentError,
    NoHistoryEmbedderForK,
    NoSearchPhraseEmbedderForK,
)
from recommender.engine.agents.rl_agent.models.history_embedder import (
    HistoryEmbedder,
    HISTORY_EMBEDDER_V1,
    HISTORY_EMBEDDER_V2,
)
from recommender.engine.agents.rl_agent.models.search_phrase_embedder import (
    SearchPhraseEmbedder,
    SEARCH_PHRASE_EMBEDDER_V1,
    SEARCH_PHRASE_EMBEDDER_V2,
)

ACTOR_V1 = "actor_v1"
ACTOR_V2 = "actor_v2"


class Actor(nn.Module):
    """Actor neural network representing a deterministic policy used by RLAgent"""

    def __init__(
        self,
        K: int,
        SE: int,
        UE: int,
        SPE: int,
        search_phrase_embedder: Optional[SearchPhraseEmbedder] = None,
        history_embedder: Optional[HistoryEmbedder] = None,
        layer_sizes: Tuple[int] = (256, 512, 256),
    ):
        """
        Args:
            K: number of services to recommend
            SE: service embedding dimension and filters embedding dimension
            UE: user embedding dimension
            SPE search phrase embedding dimension
            history_embedder: pytorch module implementing history embedding
            layer_sizes: list containing number of neurons in each hidden layer
        """
        super().__init__()

        self.K = K
        self.SE = SE

        self.history_embedder = history_embedder
        self.search_phrase_embedder = search_phrase_embedder

        self._load_models()

        layers = [nn.Linear(2 * SE + UE + SPE, layer_sizes[0]), nn.ReLU()]  # FE = SE
        layers += list(
            chain.from_iterable(
                [
                    [nn.Linear(n_size, next_n_size), nn.ReLU()]
                    for n_size, next_n_size in zip(layer_sizes, layer_sizes[1:])
                ]
            )
        )
        layers += [(nn.Linear(layer_sizes[-1], K * SE))]

        self.network = nn.Sequential(*layers)

    def forward(self, state: Tuple[(torch.Tensor,) * 4]) -> torch.Tensor:
        """
        Performs forward propagation.

        Args:
            state:
                user: Embedded user content tensor of shape [batch_size, UE]
                services_history: Services history tensor of shape [batch_size, N, SE]
                filters: Embedded filters tensor of shape [batch_size, FE]
                search_phrase: Embedded search phrase tensor of shape
                 [batch_size, X, SPE]

        Returns:
            weights: Weights tensor used for choosing action from the itemspace.
        """

        user, services_history, filters, search_phrase = state

        services_history = self.history_embedder(services_history)
        search_phrase = self.search_phrase_embedder(search_phrase)
        x = torch.cat([user, services_history, filters, search_phrase], dim=1)
        x = self.network(x)

        weights = x.reshape(-1, self.K, self.SE)
        return weights

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
        except NoSavedModuleError as no_saved_module:
            raise MissingComponentError from no_saved_module
