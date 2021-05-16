# pylint: disable-all

import torch

from recommender.engine.models.autoencoders import ServiceAutoEncoder, create_embedder
from recommender.engine.agents.panel_id_to_services_number_mapping import PANEL_ID_TO_K
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

    for K in list(PANEL_ID_TO_K.values()):
        SOH = len(Service.objects.first().tensor)
        SE = 128

        service_auto_encoder = ServiceAutoEncoder(features_dim=SOH, embedding_dim=SE)
        service_embedder = create_embedder(service_auto_encoder)

        ae = ActionEncoder(service_embedder)
        action = list(Service.objects[:K])

        encoded_action = ae(action)
        assert encoded_action.shape == torch.Size([K, SE])
