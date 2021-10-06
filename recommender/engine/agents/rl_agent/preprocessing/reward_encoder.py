# pylint: disable=missing-function-docstring, no-member, redefined-outer-name, too-few-public-methods

"""Implementation of the Reward Encoder"""

from typing import List
import torch


def reward_id_to_number(_reward_id: str) -> float:
    mapping = {
        "order": 1.0,
        "interest": 0.7,
        "mild_interest": 0.3,
        "simple_transition": 0.0,
        "unknown_transition": 0.0,
    }
    return mapping.get(_reward_id, 0.0)


def reward_ids_to_subreward(
    reward_ids: List[str], max_depth, max_steps_per_episode
) -> float:
    sub_reward = sum([reward_id_to_number(reward_id) for reward_id in reward_ids])
    sub_reward = sub_reward / (max_depth * max_steps_per_episode)
    return sub_reward


def subrewards_to_reward(rewards: List[float]) -> torch.Tensor:
    return torch.mean(torch.Tensor(rewards))


class RewardEncoder:
    """Reward Encoder"""

    def __init__(
        self,
        reward_ids_to_subreward=reward_ids_to_subreward,
        subrewards_to_reward=subrewards_to_reward,
        max_depth=10,
        max_steps_per_episode=100,
    ):
        self.reward_ids_to_subreward = reward_ids_to_subreward
        self.subrewards_to_reward = subrewards_to_reward
        self.max_depth = max_depth
        self.max_steps_per_episode = max_steps_per_episode

    def __call__(self, raw_rewards: List[List[List[str]]]) -> torch.Tensor:
        """
        Encode list of rewards (each of them is a list of reward ids) into
         scalar tensor.

        Args:
            raw_rewards: Batch of lists of lists of reward ids.

        Returns:
            reward: Scalar tensor of the total reward.
        """

        encoded_rewards = []
        for raw_reward in raw_rewards:
            rewards = list(
                map(
                    lambda x: self.reward_ids_to_subreward(
                        x, self.max_depth, self.max_steps_per_episode
                    ),
                    raw_reward,
                )
            )
            reward = self.subrewards_to_reward(rewards)
            encoded_rewards.append(reward)

        encoded_reward = torch.stack(encoded_rewards)

        return encoded_reward
