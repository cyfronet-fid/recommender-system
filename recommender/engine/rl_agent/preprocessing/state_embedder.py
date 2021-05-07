# pylint: disable=missing-module-docstring, no-member, too-few-public-methods, no-name-in-module

"""Implementation of the State Embedder"""

from typing import Tuple, List

import torch
from torch import FloatTensor

from recommender.engine.rl_agent.preprocessing.searchphrase_embedder import (
    SearchPhraseEmbedder,
)
from recommender.engine.pre_agent.models import (
    load_last_module,
    USERS_AUTOENCODER,
    SERVICES_AUTOENCODER,
)
from recommender.engine.rl_agent.preprocessing.filters_embedder import FiltersEmbedder
from recommender.models import State, Service


def _search_data_to_dict(search_data):
    """
    Transforms SearchData MongoEngine object into python dict.

    Args:
        search_data: SearchData Mongo Engine object.

    Returns:
        Python dict representation of the search data.

    """
    return dict(search_data.to_mongo())


class StateEmbedder:
    """State Embedder"""

    def __init__(
        self,
        user_embedder=None,
        service_embedder=None,
        search_phrase_embedder=None,
        filters_embedder=None,
    ):
        self.user_embedder = (
            user_embedder or load_last_module(USERS_AUTOENCODER).encoder
        )
        self.service_embedder = (
            service_embedder or load_last_module(SERVICES_AUTOENCODER).encoder
        )
        self.search_phrase_embedder = search_phrase_embedder or SearchPhraseEmbedder()
        self.filters_embedder = filters_embedder or FiltersEmbedder()

    def __call__(self, state: State) -> Tuple[torch.Tensor]:
        """
        Transform state to the tuple of tensors using appropriate embedders as
         follows:
            - state.user
                -> one_hot_tensor
                 --(user_embedder)--> user_tensor
            - state.services_history
                -> services_one_hot_tensors
                --(service_embedder)--> services_dense_tensors
                --(concat)--> services_history_tensor
            - state.last_search_data.q
                --(searchphrase_embedder)--> search_phrase_tensor
            - state.last_search_data...filters(multiple fields)
                -> one_hot_tensor
                --(filters_embedder)--> filters_tensor

        Args:
            state: State object that contains information about user, user's
             services history and searchphrase.

        Returns:
            Tuple of following tensors (in this order):
                - user of shape [UE]
                - services_history of shape [N, SE]
                - search_phrase of shape [X, SPE]
                - filters of shape [SE]

                where:
                    - UE is user content tensor embedding dim
                    - N is user clicked services history length - it is not constant
                    - SE is service content tensor embedding dim
                    - X is number of words in the search phrase embedding
                        - it is not constant
                    - SPE is search phrase tensor embedding dim
        """

        with torch.no_grad():
            user = self.user_embedder(torch.Tensor(state.user.tensor))
            services_history = self._services_history_to_tensor(state.services_history)
            search_phrase = self.search_phrase_embedder(state.last_search_data.q)
            filters = self.filters_embedder(
                _search_data_to_dict(state.last_search_data)
            )

        embedded_state = user, services_history, search_phrase, filters

        return embedded_state

    def _services_history_to_tensor(
        self, services_history: List[Service]
    ) -> FloatTensor:
        """
        Transforms list of services from user history to the services history
         tensor.

        Args:
            services_history: List of services from the user history of
             clicking and ordering.

        Returns:
            sh_dense_tensor: Services history dense, embedded tensor.
        """

        sh_one_hot_tensor = torch.stack(
            [torch.Tensor(s.tensor) for s in services_history], dim=0
        )

        sh_dense_tensor = self.service_embedder(sh_one_hot_tensor)

        return sh_dense_tensor
