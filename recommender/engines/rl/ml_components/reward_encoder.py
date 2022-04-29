# pylint: disable=missing-function-docstring, no-member, redefined-outer-name, too-few-public-methods, not-callable

"""Implementation of the Reward Encoder"""

from typing import List
import torch


class RewardEncoder:
    """Reward Encoder"""

    reward_mapping = {
        "order": 1.0,
        "interest": 0.7,
        "mild_interest": 0.3,
        "simple_transition": 0.0,
        "unknown_transition": 0.0,
        "exit": 0.0,
    }

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
            numeric_rewards = [self._symbolic_rewards_to_number(r) for r in raw_reward]
            reward = self._combine_rewards(numeric_rewards)
            encoded_rewards.append(reward)

        encoded_reward = torch.stack(encoded_rewards)

        return encoded_reward

    @classmethod
    def _symbolic_rewards_to_number(cls, reward_ids: List[str]) -> float:
        if reward_ids:
            highest_symbolic_reward = max(
                reward_ids, key=lambda x: cls.reward_mapping[x]
            )
            return cls.reward_mapping[highest_symbolic_reward]
        return 0.0

    @staticmethod
    def _combine_rewards(rewards: List[float]) -> torch.Tensor:
        # Root mean squared error
        rewards_tensor = torch.tensor(rewards)
        return (rewards_tensor**2).mean().sqrt()
