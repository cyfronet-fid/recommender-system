from typing import Tuple

import torch

from models import State


class StateEmbedder:
    def __init__(self, user_embedder=None, service_embedder=None, search_phrase_embedder=None, filters_embedder=None):
        self.user_embedder = user_embedder
        self.service_embedder = service_embedder
        self.search_phrase_embedder = search_phrase_embedder
        self.filters_embedder = filters_embedder

        # TODO: implement the rest of initialization (lazy embedders loading)
        pass

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

        # TODO: implement
        pass
