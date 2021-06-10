# pylint: disable=invalid-name, too-few-public-methods, no-member

"""Implementation of the SARS Encoder"""
from typing import Dict, Union, List

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
MASKS = "masks"
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

        actions = [SARS.action for SARS in SARSes]

        # TODO: replace (or delete) below code after merge of issue #108
        # action_tensors = self.action_encoder(actions)
        action_tensors = torch.rand(
            (len(actions), len(actions[0]), states_tensors[1].shape[2])
        )

        rewards = [SARS.reward for SARS in SARSes]
        reward_tensors = self.reward_encoder(rewards)

        sarses_batch = {
            STATE: {
                USER: states_tensors[0],
                SERVICES_HISTORY: states_tensors[1],
                MASKS: states_tensors[2],
            },
            ACTION: action_tensors,
            REWARD: reward_tensors,
            NEXT_STATE: {
                USER: next_states_tensors[0],
                SERVICES_HISTORY: next_states_tensors[1],
                MASKS: next_states_tensors[2],
            },
        }

        return sarses_batch

    def _load_components(self):
        self.state_encoder = self.state_encoder or StateEncoder()
        self.action_encoder = self.action_encoder or ActionEncoder()
        self.reward_encoder = self.reward_encoder or RewardEncoder()
