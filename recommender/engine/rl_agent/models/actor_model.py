# pylint: disable=missing-module-docstring, invalid-name, too-many-arguments, no-member

from typing import Tuple, Optional
from itertools import chain

import torch
from torch import nn

from recommender.engine.pre_agent.models import load_last_module
from recommender.engine.rl_agent.models.history_model import HistoryModel, HISTORY_MODEL
from recommender.engine.rl_agent.models.search_phrase_model import (
    SearchPhraseModel,
    SEARCH_PHRASE_MODEL,
)

ACTOR_MODEL_V1 = "actor_model_v1"
ACTOR_MODEL_V2 = "actor_model_v2"


class ActorModel(nn.Module):
    """Actor neural network representing a deterministic policy used by RLAgent"""

    def __init__(
        self,
        K: int,
        SE: int,
        UE: int,
        SPE: int,
        search_phrase_embedder: Optional[SearchPhraseModel] = None,
        history_embedder: Optional[HistoryModel] = None,
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
        self.history_embedder = history_embedder or load_last_module(HISTORY_MODEL)
        self.search_phrase_embedder = search_phrase_embedder or load_last_module(
            SEARCH_PHRASE_MODEL
        )

        layers = [nn.Linear(2 * SE + UE + SPE, layer_sizes[0]), nn.ReLU()]
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

    def forward(
        self,
        user: torch.Tensor,
        services_history: torch.Tensor,
        filters: torch.Tensor,
        search_phrase: torch.Tensor,
    ) -> torch.Tensor:
        """
        Performs forward propagation.

        Args:
            user: Embedded user content tensor of shape [UE]
            services_history: Services history tensor of shape [N, SE]
            filters: Embedded filters tensor of shape [FE]
            search_phrase: Embedded search phrase tensor of shape [X, SPE]

        """

        embedded_services_history = self.history_embedder(services_history)
        embedded_search_phrase = self.search_phrase_embedder(search_phrase)
        x = torch.cat(
            [embedded_services_history, user, filters, embedded_search_phrase], dim=1
        )
        x = self.network(x)

        weights = x.reshape(-1, self.K, self.SE)
        return weights
