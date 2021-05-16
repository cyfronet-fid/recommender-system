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
from recommender.engine.agents.rl_agent.preprocessing.state_encoder import StateEncoder
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


@pytest.fixture
def parameters():
    UE = 32
    SE = 128
    SPE = 100
    N = 10
    X = 5
    return UE, SE, SPE, N, X


def test_search_phrase_encoder(mongo):
    q = "Let's test the search phrase"
    search_phrase_encoder = SearchPhraseEncoder()
    search_phrase_encoder(q)


def test_filters_encoder(mongo, parameters):
    SE = parameters[2]
    state = StateFactory(services_history=ServiceFactory.create_batch(10))
    services_transformer = precalculate_tensors(
        Service.objects, create_transformer(SERVICES)
    )
    state.reload()

    SOH = len(Service.objects[0].tensor)
    service_embedder = create_embedder(
        ServiceAutoEncoder(features_dim=SOH, embedding_dim=SE)
    )

    filters_encoder = FiltersEncoder(
        service_transformer=services_transformer,
        service_embedder=service_embedder,
    )

    encoded_filters = filters_encoder(state.last_search_data)

    assert type(encoded_filters) == torch.Tensor
    assert encoded_filters.shape == torch.Size([SE])


def test_state_encoder(mongo, mocker, parameters):
    UE, SE, SPE, N, X = parameters

    populate_users_and_services(
        common_services_number=9,
        no_one_services_number=9,
        users_number=5,
        k_common_services_min=5,
        k_common_services_max=7,
    )

    precalc_users_and_service_tensors()

    UOH = len(User.objects[0].tensor)
    SOH = len(Service.objects[0].tensor)

    service_transformer = load_last_transformer(SERVICES)

    user_embedder = create_embedder(UserAutoEncoder(features_dim=UOH, embedding_dim=UE))

    service_embedder = create_embedder(
        ServiceAutoEncoder(features_dim=SOH, embedding_dim=SE)
    )

    search_phrase_encoder = SearchPhraseEncoder(word_to_vector=None)

    filters_encoder = FiltersEncoder(
        service_transformer=service_transformer, service_embedder=service_embedder
    )

    state_encoder = StateEncoder(
        user_embedder=user_embedder,
        service_embedder=service_embedder,
        search_phrase_encoder=search_phrase_encoder,
        filters_encoder=filters_encoder,
    )

    services_history = list(Service.objects)[:N]
    user = User.objects.first()
    state = StateFactory(
        user=user,
        services_history=services_history,
    )

    encoded_state = state_encoder(state)

    assert type(encoded_state) == tuple
    assert len(encoded_state) == 4
    for encoding in encoded_state:
        assert type(encoding) == torch.Tensor

    assert encoded_state[0].shape == torch.Size([UE])
    assert encoded_state[1].shape == torch.Size([N, SE])
    assert encoded_state[2].shape == torch.Size([SE])
    assert encoded_state[3].shape[1] == SPE
