# pylint: disable-all

import torch

from recommender.engine.agents.rl_agent.preprocessing.reward_encoder import (
    RewardEncoder,
)
from tests.factories.sars import SarsFactory


def test_reward_encoder_proper_shape(mongo):
    SARS_K_2 = SarsFactory(K_2=True)
    SARS_K_3 = SarsFactory(K_3=True)

    for SARS in (SARS_K_2, SARS_K_3):
        reward_encoder = RewardEncoder()
        reward = reward_encoder(SARS.reward)

        assert isinstance(reward, torch.Tensor)
        assert reward.shape == torch.Size([])
