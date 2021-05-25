# pylint: disable=missing-function-docstring, no-member, redefined-outer-name, too-few-public-methods

"""Implementation of the Reward Encoder"""

import random
from typing import List
import torch


# TODO: these function should be reimplemented according to reward mapping algorithm
def reward_id_to_number(reward_id: str) -> float:
    return random.random()


def reward_ids_to_subreward(reward_ids: List[str]) -> float:
    sub_reward = sum([reward_id_to_number(reward_id) for reward_id in reward_ids])
    return sub_reward


def subrewards_to_reward(rewards: List[float]) -> torch.Tensor:
    return torch.mean(torch.Tensor(rewards))


class RewardEncoder:
    """Reward Encoder"""

    def __init__(
        self,
        reward_ids_to_subreward=reward_ids_to_subreward,
        subrewards_to_reward=subrewards_to_reward,
    ):
        self.reward_ids_to_subreward = reward_ids_to_subreward
        self.subrewards_to_reward = subrewards_to_reward

    def __call__(self, raw_reward: List[List[str]]) -> torch.Tensor:
        """
        Encode list of rewards (each of them is a list of reward ids) into
         scalar tensor.

        Args:
            raw_reward: List of lists of reward ids.

        Returns:
            reward: Scalar tensor of the total reward.
        """

        rewards = list(map(self.reward_ids_to_subreward, raw_reward))
        reward = self.subrewards_to_reward(rewards)

        return reward
