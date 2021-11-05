# pylint: disable=invalid-name, too-few-public-methods, no-member, too-many-arguments

"""Implementation of the SARS Encoder"""

from typing import Dict, Union, List

import torch

from recommender.engines.autoencoders.ml_components.embedder import Embedder
from recommender.engines.rl.ml_components.encoders.reward_encoder import RewardEncoder
from recommender.engines.rl.ml_components.services2weights import Services2Weights
from recommender.engines.rl.ml_components.encoders.state_encoder import StateEncoder
from recommender.models import Sars

STATE = "state"
NEXT_STATE = "next_state"
USER = "user"
SERVICES_HISTORY = "services_history"
MASK = "mask"
ACTION = "action"
REWARD = "reward"


class SarsEncoder:
    """SARS Encoder"""

    def __init__(
        self,
        user_embedder: Embedder,
        service_embedder: Embedder,
        use_cached_embeddings: bool = True,
        save_cached_embeddings: bool = False,
    ):
        self.use_cached_embeddings = use_cached_embeddings
        self.save_cached_embeddings = save_cached_embeddings
        self.state_encoder = StateEncoder(
            user_embedder,
            service_embedder,
            use_cached_embeddings=self.use_cached_embeddings,
            save_cached_embeddings=self.save_cached_embeddings,
        )
        self.reward_encoder = RewardEncoder()
        self.services2weights = Services2Weights(
            service_embedder,
            use_cached_embeddings=self.use_cached_embeddings,
            save_cached_embeddings=self.save_cached_embeddings,
        )

    def __call__(
        self, SARSes: List[Sars]
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Encode SARS MongoEngine object into training example.

        Args:
            SARSes: List of Mongoengine Sars objects.

        Returns:
            sarses_batch: Dict of named tensors.
        """

        states = [SARS.state for SARS in SARSes]
        next_states = [SARS.next_state for SARS in SARSes]
        all_states = states + next_states
        all_states_tensors = self.state_encoder(all_states)

        states_tensors = tuple(t[: len(states)] for t in all_states_tensors)
        next_states_tensors = tuple(t[len(states) :] for t in all_states_tensors)

        service_ids = [
            [s.id for s in SARS.action] for SARS in SARSes
        ]  # "actions" from DB

        weight_tensors = self.services2weights(service_ids)  # actual action

        rewards = [SARS.reward for SARS in SARSes]
        reward_tensors = self.reward_encoder(rewards)

        sarses_batch = {
            STATE: {
                USER: states_tensors[0],
                SERVICES_HISTORY: states_tensors[1],
                MASK: states_tensors[2],
            },
            ACTION: weight_tensors,
            REWARD: reward_tensors,
            NEXT_STATE: {
                USER: next_states_tensors[0],
                SERVICES_HISTORY: next_states_tensors[1],
                MASK: next_states_tensors[2],
            },
        }

        return sarses_batch
