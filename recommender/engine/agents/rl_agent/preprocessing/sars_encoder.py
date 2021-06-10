# pylint: disable=invalid-name, too-few-public-methods, no-member

"""Implementation of the SARS Encoder"""

from typing import Dict, Union, List

import torch

from recommender.engine.agents.rl_agent.action_inverter import ActionInverter
from recommender.models import Sars
from recommender.engine.agents.rl_agent.preprocessing.reward_encoder import (
    RewardEncoder,
)
from recommender.engine.agents.rl_agent.preprocessing.state_encoder import StateEncoder

STATE = "state"
NEXT_STATE = "next_state"
USER = "user"
SERVICES_HISTORY = "services_history"
MASK = "mask"
ACTION = "action"
REWARD = "reward"


class SarsEncoder:
    """SARS Encoder"""

    def __init__(self, state_encoder=None, reward_encoder=None, action_inverter=None):
        self.state_encoder = state_encoder
        self.reward_encoder = reward_encoder
        self.action_inverter = action_inverter

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

        actions = [[s.id for s in SARS.action] for SARS in SARSes]

        action_tensors = self.action_inverter(actions)

        rewards = [SARS.reward for SARS in SARSes]
        reward_tensors = self.reward_encoder(rewards)

        sarses_batch = {
            STATE: {
                USER: states_tensors[0],
                SERVICES_HISTORY: states_tensors[1],
                MASK: states_tensors[2],
            },
            ACTION: action_tensors,
            REWARD: reward_tensors,
            NEXT_STATE: {
                USER: next_states_tensors[0],
                SERVICES_HISTORY: next_states_tensors[1],
                MASK: next_states_tensors[2],
            },
        }

        return sarses_batch

    def _load_components(self):
        self.state_encoder = self.state_encoder or StateEncoder()
        self.reward_encoder = self.reward_encoder or RewardEncoder()
        self.action_inverter = self.action_inverter or ActionInverter()
