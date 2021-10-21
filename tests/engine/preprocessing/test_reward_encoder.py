# pylint: disable-all
import torch
import pytest

from recommender.engines.rl.ml_components.reward_encoder import (
    RewardEncoder,
)


@pytest.fixture
def proper_rewards():
    raw_rewards = [
        [
            ["order", "interest"],
            ["order", "mild_interest"],
            ["simple_transition", "exit"],
        ],
        [
            ["interest", "simple_transition"],
            ["mild_interest", "exit", "simple_transition"],
            ["exit"],
        ],
    ]
    desired_output = torch.Tensor([0.8165, 0.4397])
    return raw_rewards, desired_output


def test_reward_encoder_proper_shape(proper_rewards):
    raw_rewards, desired_output = proper_rewards
    reward_encoder = RewardEncoder()
    rewards = reward_encoder(raw_rewards)
    assert torch.allclose(rewards, desired_output, atol=10e-4)
