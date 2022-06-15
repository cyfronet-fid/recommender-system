# pylint: disable-all

import torch

from recommender.engines.nlp_embedders.embedders import (
    Users2tensorsEmbedder,
    Services2tensorsEmbedder,
)
from recommender.engines.rl.ml_components.state_encoder import StateEncoder
from recommender.models import User, Service

from tests.factories.marketplace import ServiceFactory
from tests.factories.search_data import SearchDataFactory
from tests.factories.state import StateFactory


def test_state_encoder(mongo):
    # prepare data
    UE = Users2tensorsEmbedder().embedding_dim
    SE = Services2tensorsEmbedder().embedding_dim

    services_histories = [
        ServiceFactory.create_batch(3),
        [],
        ServiceFactory.create_batch(5),
    ]

    B = len(services_histories)
    max_N = max(len(services_history) for services_history in services_histories)

    states = []
    for services_history in services_histories:
        state = StateFactory(
            services_history=services_history, search_data=SearchDataFactory(q=None)
        )
        states.append(state)

    for i in range(len(services_histories)):
        for j in range(len(services_histories[i])):
            services_histories[i][j].reload()

    state_encoder = StateEncoder()

    encoded_state = state_encoder(states)

    assert type(encoded_state) == tuple
    assert len(encoded_state) == 3
    for encoding in encoded_state:
        assert type(encoding) == torch.Tensor

    users_batch, service_histories_batch, masks_batch = encoded_state
    print(masks_batch.shape)

    assert users_batch.shape == torch.Size([B, UE])

    assert service_histories_batch.shape == torch.Size([B, max_N, SE])

    assert len(masks_batch.shape) == 2
    assert masks_batch.shape[0] == B
    assert masks_batch.shape[1] == len(Service.objects)
