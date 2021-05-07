# pylint: disable-all

import torch

from recommender.engine.rl_agent.preprocessing.state_embedder import StateEmbedder
from recommender.engine.pre_agent.preprocessing import precalculate_tensors, create_transformer, SERVICES, USERS
from recommender.engine.rl_agent.preprocessing.filters_embedder import FiltersEmbedder
from recommender.models import Service, User
from recommender.engine.rl_agent.preprocessing.searchphrase_embedder import SearchPhraseEmbedder
from tests.factories.marketplace import ServiceFactory
from tests.factories.state import StateFactory


def test_search_phrase_embedder(mongo):
    q = "Let's test the search phrase"
    search_phrase_embedder = SearchPhraseEmbedder()
    search_phrase_embedder(q)


def test_filters_embedder(mongo):
    state = StateFactory(
        services_history=ServiceFactory.create_batch(10)
    )

    services_transformer = precalculate_tensors(
        Service.objects,
        create_transformer(SERVICES)
    )

    precalculate_tensors(
        User.objects,
        create_transformer(USERS)
    )

    state.reload()

    SE = 128
    service_embedder = lambda _: torch.rand(SE)

    filters_embedder = FiltersEmbedder(
        services_transformer=services_transformer,
        service_embedder=service_embedder
    )

    search_data = dict(state.last_search_data.to_mongo())
    embedded_filters = filters_embedder(search_data)

    assert type(embedded_filters) == torch.Tensor
    assert embedded_filters.shape == torch.Size([SE])


def test_state_embedder(mongo, mocker):
    UE = 32
    SE = 128
    SPE = 100
    N = 10
    X = 5

    state = StateFactory(
        services_history=ServiceFactory.create_batch(N)
    )

    precalculate_tensors(
        Service.objects,
        create_transformer(SERVICES)
    )

    precalculate_tensors(
        User.objects,
        create_transformer(USERS)
    )

    state.reload()

    user_embedder = lambda _: torch.rand(UE)
    service_embedder = lambda t: torch.rand(t.shape[0], SE)

    search_phrase_embedder_mock = mocker.patch(
        "recommender.engine.rl_agent.preprocessing.searchphrase_embedder.SearchPhraseEmbedder.__call__"
    )
    search_phrase_embedder_mock.return_value = torch.stack([torch.rand(SPE) for _ in range(X)])
    search_phrase_embedder = SearchPhraseEmbedder()

    filters_embedder = lambda _: torch.rand(SE)
    filters_embedder_mock = mocker.patch(
        "recommender.engine.rl_agent.preprocessing.filters_embedder.FiltersEmbedder.__call__"
    )
    filters_embedder_mock.return_value = torch.rand(SE)

    state_embedder = StateEmbedder(
        user_embedder=user_embedder,
        service_embedder=service_embedder,
        search_phrase_embedder=search_phrase_embedder,
        filters_embedder=filters_embedder
    )

    embedded_state = state_embedder(state)

    assert type(embedded_state) == tuple
    assert len(embedded_state) == 4
    for embedding in embedded_state:
        assert type(embedding) == torch.Tensor

    assert embedded_state[0].shape == torch.Size([UE])
    assert embedded_state[1].shape == torch.Size([N, SE])
    assert embedded_state[2].shape == torch.Size([X, SPE])
    assert embedded_state[3].shape == torch.Size([SE])
