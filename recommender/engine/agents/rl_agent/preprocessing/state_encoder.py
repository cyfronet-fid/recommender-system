# pylint: disable=missing-module-docstring, no-member, too-few-public-methods, no-name-in-module

"""Implementation of the State Encoder"""

from typing import Tuple, Optional

import torch

from recommender.errors import MissingComponentError
from recommender.engine.agents.rl_agent.utils import (
    use_service_embedder,
    use_user_embedder,
)
from recommender.engine.agents.rl_agent.preprocessing.searchphrase_encoder import (
    SearchPhraseEncoder,
)
from recommender.engine.models.autoencoders import (
    USERS_AUTOENCODER,
    SERVICES_AUTOENCODER,
    create_embedder,
)
from recommender.engine.utils import load_last_module, NoSavedModuleError
from recommender.engine.agents.rl_agent.preprocessing.filters_encoder import (
    FiltersEncoder,
)
from recommender.models import State


class StateEncoder:
    """State Encoder"""

    def __init__(
        self,
        user_embedder: Optional[torch.nn.Module] = None,
        service_embedder: Optional[torch.nn.Module] = None,
        search_phrase_encoder: Optional[SearchPhraseEncoder] = None,
        filters_encoder: Optional[FiltersEncoder] = None,
    ):
        self.user_embedder = user_embedder
        self.service_embedder = service_embedder
        self.search_phrase_encoder = search_phrase_encoder
        self.filters_encoder = filters_encoder

        self._load_components()

    def __call__(self, state: State) -> Tuple[(torch.Tensor,) * 4]:
        """
        Encode state to the tuple of tensors using appropriate encoders and embedders as
         follows:
            - state.user
                -> one_hot_tensor
                 --(user_embedder)--> user_tensor
            - state.services_history
                -> services_one_hot_tensors
                --(service_embedder)--> services_dense_tensors
                --(concat)--> services_history_tensor
            - state.last_search_data...filters(multiple fields)
                -> one_hot_tensor
                --(filters_encoder)--> filters_tensor
            - state.last_search_data.q
                --(search_phrase_encoder)--> search_phrase_tensor

        Args:
            state: State object that contains information about user, user's
             services history and searchphrase.

        Returns:
            Tuple of following tensors (in this order):
                - user of shape [UE]
                - services_history of shape [N, SE]
                - filters of shape [SE]
                - search_phrase of shape [X, SPE]


                where:
                    - UE is user content tensor embedding dim
                    - N is user clicked services history length - it is not constant
                    - SE is service content tensor embedding dim
                    - X is number of words in the search phrase embedding
                        - it is not constant
                    - SPE is search phrase tensor embedding dim
        """

        with torch.no_grad():
            user = use_user_embedder(state.user, self.user_embedder)
            services_history, _ = use_service_embedder(
                state.services_history, self.service_embedder
            )
            filters = self.filters_encoder(state.last_search_data)
            search_phrase = self.search_phrase_encoder(state.last_search_data.q)

        encoded_state = user, services_history, filters, search_phrase

        return encoded_state

    def _load_components(self):
        try:
            self.user_embedder = self.user_embedder or create_embedder(
                load_last_module(USERS_AUTOENCODER)
            )
            self.service_embedder = self.service_embedder or create_embedder(
                load_last_module(SERVICES_AUTOENCODER)
            )
        except NoSavedModuleError as no_saved_module:
            raise MissingComponentError from no_saved_module

        self.user_embedder.eval()
        self.service_embedder.eval()

        self.search_phrase_encoder = self.search_phrase_encoder or SearchPhraseEncoder()
        self.filters_encoder = self.filters_encoder or FiltersEncoder()
