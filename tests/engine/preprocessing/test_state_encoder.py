# pylint: disable-all

import torch

from recommender.engine.models.autoencoders import AutoEncoder
from recommender.engines.autoencoders.ml_components.embedder import Embedder
from recommender.engines.autoencoders.training.data_preparation_step import (
    precalc_users_and_service_tensors,
)
from recommender.models import User, Service
from recommender.engine.agents.rl_agent.preprocessing.state_encoder import StateEncoder

from tests.factories.marketplace import ServiceFactory
from tests.factories.search_data import SearchDataFactory
from tests.factories.state import StateFactory


def test_state_encoder(mongo):
    # prepare data
    UE = 32
    SE = 128

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

    precalc_users_and_service_tensors()

    for state in states:
        state.reload()

    for i in range(len(services_histories)):
        for j in range(len(services_histories[i])):
            services_histories[i][j].reload()

    UOH = len(User.objects[0].one_hot_tensor)
    SOH = len(Service.objects[0].one_hot_tensor)

    # prepare state encoder
    user_autoencoder = AutoEncoder(features_dim=UOH, embedding_dim=UE)
    user_autoencoder.eval()
    service_autoencoder = AutoEncoder(features_dim=SOH, embedding_dim=SE)
    service_autoencoder.eval()
    user_embedder = Embedder(user_autoencoder)
    service_embedder = Embedder(service_autoencoder)

    state_encoder = StateEncoder(
        user_embedder=user_embedder,
        service_embedder=service_embedder,
    )

    encoded_state = state_encoder(states)

    assert type(encoded_state) == tuple
    assert len(encoded_state) == 3
    for encoding in encoded_state:
        assert type(encoding) == torch.Tensor

    users_batch, service_histories_batch, masks_batch = encoded_state
    print(masks_batch.shape)

    assert users_batch.shape == torch.Size([B, UE])

    assert service_histories_batch.shape == torch.Size([B, max_N, SE])

    for i in range(len(services_histories)):
        for j in range(len(services_histories[i])):
            original_oh_tensor = torch.Tensor(
                services_histories[i][j].one_hot_tensor
            ).unsqueeze(0)
            original_embedding = service_autoencoder.encoder(
                original_oh_tensor
            ).squeeze(0)
            tested_embedding = service_histories_batch[i][j]
            assert all(
                torch.isclose(
                    original_embedding, tested_embedding, rtol=1e-2, atol=1e-2
                )
            )

    assert len(masks_batch.shape) == 2
    assert masks_batch.shape[0] == B
    assert masks_batch.shape[1] == len(Service.objects)
