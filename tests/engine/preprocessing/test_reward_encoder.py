# pylint: disable-all

import torch

from recommender.engine.agents.rl_agent.preprocessing.reward_encoder import (
    RewardEncoder,
)
from tests.factories.sars import SarsFactory


def test_reward_encoder_proper_shape(mongo):
    B = 3
    SARSes_K_2 = SarsFactory.create_batch(B, K_2=True)
    SARSes_K_3 = SarsFactory.create_batch(B, K_3=True)

    for SARSes in (SARSes_K_2, SARSes_K_3):
        reward_encoder = RewardEncoder()
        raw_rewards = [SARS.reward for SARS in SARSes]
        rewards = reward_encoder(raw_rewards)

        assert isinstance(rewards, torch.Tensor)
        assert rewards.shape == torch.Size([B])
