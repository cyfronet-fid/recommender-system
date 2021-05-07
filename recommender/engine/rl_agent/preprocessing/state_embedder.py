from typing import Tuple

import torch

from engine.pre_agent.models import load_last_module, USERS_AUTOENCODER, SERVICES_AUTOENCODER
from models import State


class StateEmbedder:
    def __init__(self, user_embedder=None, service_embedder=None, search_phrase_embedder=None, filters_embedder=None):
        if user_embedder is not None:
            self.user_embedder = user_embedder
        else:
            self.user_embedder = load_last_module(USERS_AUTOENCODER).encoder

        if service_embedder is not None:
            self.service_embedder = service_embedder
        else:
            self.service_embedder = load_last_module(SERVICES_AUTOENCODER).encoder

        if search_phrase_embedder is not None:
            self.search_phrase_embedder = search_phrase_embedder
        else:
            self.search_phrase_embedder = SearchPhraseEmbedder()

        if filters_embedder is not None:
            self.filters_embedder = filters_embedder
        else:
            self.filters_embedder = FilterEmbedder()


    def __call__(self, state: State) -> Tuple[torch.Tensor]:
        """
        Transform state to the tuple of tensors using appropriate embedders as
         follows:
            - state.user -> one_hot_tensor --(user_embedder)--> user_tensor
            - state.user.accessed_services and clicks information??? -> services_one_hot_tensors --(service_embedder)--> services_dense_tensors --(concat)--> services_history_tensor
            - state.last_search_data...filters(multiple fields) -> one_hot_tensor --(filters_embedder)--> filters_tensor
            - state.last_search_data.q --(searchphrase_embedder)--> search_phrase_tensor

        Args:
            state: State object that contains information about user, user's services history and searchphrase.

        Returns:
            Tuple of following tensors (in this order):
                - user_tensor of shape [UE]
                - services_history_tensor of shape [N, SE]
                - filters_tensor of shape [FE]
                - search_phrase_tensor of shape [SPE]
                where:
                    - UE is user content tensor embedding dim
                    - N is user clicked services history length
                    - SE is service content tensor embedding dim
                    - FE is filters tensor embedding dim
                    - SPE is search phrase tensor embedding dim
        """

        user_dense_tensor = self.user_embedder(state.user.tensor)


        pass
