# pylint: disable=missing-module-docstring, no-member, too-few-public-methods
# pylint: disable=no-name-in-module, invalid-name, too-many-locals
# pylint: disable=too-many-branches

"""Implementation of the State Encoder"""
from typing import Tuple, Optional, List

import torch
from torch.nn.utils.rnn import pad_sequence

from recommender.engine.agents.rl_agent.preprocessing.search_data_encoder import (
    SearchDataEncoder,
)
from recommender.models import Service
from recommender.errors import MissingComponentError
from recommender.engine.agents.rl_agent.utils import (
    use_service_embedder,
    use_user_embedder,
)
from recommender.engine.models.autoencoders import (
    USERS_AUTOENCODER,
    SERVICES_AUTOENCODER,
    create_embedder,
)
from recommender.engine.utils import load_last_module, NoSavedModuleError
from recommender.models import State


class StateEncoder:
    """State Encoder"""

    def __init__(
        self,
        user_embedder: Optional[torch.nn.Module] = None,
        service_embedder: Optional[torch.nn.Module] = None,
        search_data_encoder: SearchDataEncoder = None,
    ):
        self.user_embedder = user_embedder
        self.service_embedder = service_embedder
        self.search_data_encoder = search_data_encoder

        self._load_components()

    def __call__(self, states: List[State]) -> Tuple[(torch.Tensor,) * 3]:
        """
        Encode given states to the tuple of tensors using appropriate encoders
         and embedders as follows:
            - state.user
                -> one_hot_tensor
                 --(user_embedder)--> users_dense_tensor
            - state.services_history
                -> services_one_hot_tensors
                --(service_embedder)--> services_dense_tensors
                --(concat)--> services_histories_tensor
            - state.search_data
                --(search_data_encoder)--> search_data_tensor (mask)


        It makes batches from parts of states whenever it is possible to
         optimally use embedders.

        Args:
            states: List of state objects. Each of them contains information
             about user, user's services history and search data.

        Returns:
            Tuple of following tensors (in this order):
                - user of shape [B, UE]
                - service_histories_batch of shape [B, N, SE]
                - search_data_mask of shape [B, I]

                where:
                    - B is the batch_size and it is equal to len(states)
                    - UE is user content tensor embedding dim
                    - N is user clicked services history length - it is not
                     constant
                    - SE is service content tensor embedding dim
                    - K is the first mask dim
                    - I is equal to itemspace size
        """

        users = [state.user for state in states]
        services_histories = [state.services_history for state in states]
        search_data_list = [state.search_data for state in states]
        with torch.no_grad():
            users_batch = use_user_embedder(users, self.user_embedder)
            service_histories_batch = self._create_service_histories_batch(
                services_histories
            )
            search_data_masks_batch = self.search_data_encoder(users, search_data_list)

        encoded_states = users_batch, service_histories_batch, search_data_masks_batch

        return encoded_states

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

        self.search_data_encoder = self.search_data_encoder or SearchDataEncoder()

    def _create_service_histories_batch(
        self, services_histories: List[List[Service]]
    ) -> torch.Tensor:
        """
        Create services history batch out of list of service lists. Handle
         empty history.

        Args:
            services_histories: List of Service lists

        Returns:
            services_history: Tensor of services histories of shape
             [B, maxN, SE], where:
                -> B is len(services_histories),
                -> maxN is the max services history length (among all given
                 services histories),
                -> SE is Service Embedding dim.
        """

        states_number = len(services_histories)
        services = []
        start_end_indices = []
        for state_idx in range(states_number):
            cur_state_history = services_histories[state_idx]
            cur_history_len = len(cur_state_history)
            if cur_history_len > 0:
                start_idx = len(services)
                for service_idx in range(cur_history_len):
                    services.append(cur_state_history[service_idx])
                end_idx = start_idx + cur_history_len
                start_end_indices.append((start_idx, end_idx))
            else:
                start_end_indices.append(None)

        if services:
            service_tensors, _ = use_service_embedder(services, self.service_embedder)

        sequences = []
        for state_idx in range(states_number):
            indices = start_end_indices[state_idx]
            if indices is not None:
                start_idx = indices[0]
                end_idx = indices[1]
                services_history_tensor = service_tensors[start_idx:end_idx]
                sequences.append(services_history_tensor)
            else:
                N = 1
                SE = self.service_embedder[-1].out_features
                services_history_tensor = torch.zeros((N, SE))
                sequences.append(services_history_tensor)

        service_histories_batch = pad_sequence(sequences, batch_first=True)

        return service_histories_batch
