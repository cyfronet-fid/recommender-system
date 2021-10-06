# pylint: disable-all

import torch

from recommender.engine.agents.rl_agent.utils import embedded_tensors_exist
from tests.factories.marketplace import UserFactory


def test_embedded_tensors_exist(mongo):
    UE = 32
    users = UserFactory.create_batch(3)
    assert embedded_tensors_exist(users) is False

    for user in users:
        user.embedded_tensor = torch.rand(UE).tolist()
        user.save()

    assert embedded_tensors_exist(users) is True

    users[1].embedded_tensor = []
    users[1].save()

    assert embedded_tensors_exist(users) is False

