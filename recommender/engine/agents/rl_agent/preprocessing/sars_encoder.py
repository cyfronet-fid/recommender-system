# pylint: disable=invalid-name, too-few-public-methods

"""Implementation of the SARS Encoder"""
from typing import Dict, Union

import torch

from recommender.models import Sars
from recommender.engine.agents.rl_agent.preprocessing.action_encoder import (
    ActionEncoder,
)
from recommender.engine.agents.rl_agent.preprocessing.reward_encoder import (
    RewardEncoder,
)
from recommender.engine.agents.rl_agent.preprocessing.state_encoder import StateEncoder

STATE = "state"
NEXT_STATE = "next_state"
USER = "user"
SERVICES_HISTORY = "services_history"
FILTERS = "filters"
SEARCH_PHRASE = "search_phrase"
ACTION = "action"
REWARD = "reward"


class SarsEncoder:
    """SARS Encoder"""

    def __init__(
        self,
        state_encoder=None,
        action_encoder=None,
        reward_encoder=None,
    ):
        self.state_encoder = state_encoder
        self.action_encoder = action_encoder
        self.reward_encoder = reward_encoder

        self._load_components()

    def __call__(
        self, SARS: Sars
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Encode SARS MongoEngine object into training example.

        Args:
            SARS: Mongoengine Sars object.

        Returns:
            sars_example: Dict of named tensors.
        """
        state_tensors = self.state_encoder(SARS.state)
        action_tensor = self.action_encoder(SARS.action)
        reward_tensor = self.reward_encoder(SARS.reward)
        next_state_tensors = self.state_encoder(SARS.next_state)

        sars_example = {
            STATE: {
                USER: state_tensors[0],
                SERVICES_HISTORY: state_tensors[1],
                FILTERS: state_tensors[2],
                SEARCH_PHRASE: state_tensors[3],
            },
            ACTION: action_tensor,
            REWARD: reward_tensor,
            NEXT_STATE: {
                USER: next_state_tensors[0],
                SERVICES_HISTORY: next_state_tensors[1],
                FILTERS: next_state_tensors[2],
                SEARCH_PHRASE: next_state_tensors[3],
            },
        }

        return sars_example

    def _load_components(self):
        self.state_encoder = self.state_encoder or StateEncoder()
        self.action_encoder = self.action_encoder or ActionEncoder()
        self.reward_encoder = self.reward_encoder or RewardEncoder()
