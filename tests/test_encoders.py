# pylint: disable-all

import pytest
import torch

from recommender.engine.preprocessing import load_last_transformer
from recommender.models import User, Service
from recommender.engine.preprocessing import precalc_users_and_service_tensors
from recommender.engine.models.autoencoders import (
    create_embedder,
    ServiceAutoEncoder,
    UserAutoEncoder,
)
from recommender.engine.agents.rl_agent.preprocessing.searchphrase_encoder import (
    SearchPhraseEncoder,
)
from recommender.engine.agents.rl_agent.preprocessing.state_encoder import (
    StateEncoder,
    MaskEncoder,
)
from recommender.engine.preprocessing import (
    precalculate_tensors,
    create_transformer,
    SERVICES,
)
from recommender.engine.agents.rl_agent.preprocessing.filters_encoder import (
    FiltersEncoder,
)
from tests.factories.marketplace import ServiceFactory
from tests.factories.populate_database import populate_users_and_services
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
            services_history=services_history,
        )
        states.append(state)

    precalc_users_and_service_tensors()

    for state in states:
        state.reload()

    for i in range(len(services_histories)):
        for j in range(len(services_histories[i])):
            services_histories[i][j].reload()

    UOH = len(User.objects[0].tensor)
    SOH = len(Service.objects[0].tensor)

    # prepare state encoder
    user_embedder = create_embedder(UserAutoEncoder(features_dim=UOH, embedding_dim=UE))
    user_embedder.eval()

    service_embedder = create_embedder(
        ServiceAutoEncoder(features_dim=SOH, embedding_dim=SE)
    )
    service_embedder.eval()

    mask_encoder = MaskEncoder()

    state_encoder = StateEncoder(
        user_embedder=user_embedder,
        service_embedder=service_embedder,
        mask_encoder=mask_encoder,
    )

    encoded_state = state_encoder(states)

    assert type(encoded_state) == tuple
    assert len(encoded_state) == 3
    for encoding in encoded_state:
        assert type(encoding) == torch.Tensor

    users_batch, service_histories_batch, masks_batch = encoded_state

    assert users_batch.shape == torch.Size([B, UE])

    assert service_histories_batch.shape == torch.Size([B, max_N, SE])

    for i in range(len(services_histories)):
        for j in range(len(services_histories[i])):
            original_oh_tensor = torch.Tensor(
                services_histories[i][j].tensor
            ).unsqueeze(0)
            original_embedding = service_embedder(original_oh_tensor).squeeze(0)
            tested_embedding = service_histories_batch[i][j]
            assert all(
                torch.isclose(
                    original_embedding, tested_embedding, rtol=1e-2, atol=1e-2
                )
            )

    assert len(masks_batch.shape) == 3
    assert masks_batch.shape[0] == B
