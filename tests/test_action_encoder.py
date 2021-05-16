# pylint: disable-all

import torch

from recommender.engine.preprocessing import precalc_users_and_service_tensors
from recommender.models import Service
from recommender.engine.agents.rl_agent.preprocessing.action_encoder import (
    ActionEncoder,
)
from tests.factories.populate_database import populate_users_and_services


def test_action_encoder_proper_shape(mongo):
    populate_users_and_services(
        common_services_number=4,
        no_one_services_number=1,
        users_number=4,
        k_common_services_min=1,
        k_common_services_max=3,
    )

    precalc_users_and_service_tensors()

    for K in (2, 3):
        SOH = len(Service.objects.first().tensor)
        SE = 128

        services_embedder = torch.nn.Linear(SOH, SE)
        ae = ActionEncoder(services_embedder)
        action = list(Service.objects[:K])

        embedded_action = ae(action)
        assert embedded_action.shape == torch.Size([K, SE])
