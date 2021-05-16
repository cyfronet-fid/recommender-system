# pylint: disable=too-few-public-methods

"""Replay Buffer Generator implementation"""
from typing import Optional

import torch

from recommender.engine.agents.rl_agent.preprocessing.action_encoder import (
    ActionEncoder,
)
from recommender.engine.agents.rl_agent.preprocessing.state_encoder import StateEncoder


class ReplayBufferGenerator:
    # WARNING: check if services history is generated (by SARSes generator)
    # from accessed services AND clicked services!
    """It generates a replay buffer - dataset for the RL Agent"""

    def __init__(
        self,
        state_encoder: Optional[StateEncoder] = None,
        action_encoder: Optional[ActionEncoder] = None,
    ):
        # TODO: implement the rest of initialization including the lazy loading
        pass

    def __call__(self) -> torch.utils.data.Dataset:
        """
        Generates a pytorch dataset.

        Returns:
            RL-Agent Dataset.

        """

        # TODO: implement
        pass
