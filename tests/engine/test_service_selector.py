# pylint: disable-all
import numpy as np
import pandas as pd
import pytest
import torch

from recommender.engines.autoencoders.ml_components.autoencoder import AutoEncoder
from recommender.engines.autoencoders.ml_components.embedder import Embedder
from recommender.engines.rl.ml_components.service_selector import ServiceSelector
from recommender.errors import InsufficientRecommendationSpaceError
from tests.factories.marketplace import ServiceFactory


@pytest.fixture
def proper_parameters():
    FEATURES_DIM = 10
    K = 2
    SE = 4
    return FEATURES_DIM, K, SE


@pytest.fixture
def services(mongo, proper_parameters):
    FEATURES_DIM = proper_parameters[0]

    return [
        ServiceFactory(id=2, one_hot_tensor=torch.randint(2, (FEATURES_DIM,))),
        ServiceFactory(id=8, one_hot_tensor=torch.randint(2, (FEATURES_DIM,))),
        ServiceFactory(id=4, one_hot_tensor=torch.randint(2, (FEATURES_DIM,))),
        ServiceFactory(id=6, one_hot_tensor=torch.randint(2, (FEATURES_DIM,))),
    ]


@pytest.fixture
def service_embeddings():
    return torch.Tensor(
        [
            [0.9, 0.5, 1.6, -2.0],  # id=2
            [0.8, -1.5, 3.0, 1.0],  # id=4
            [-0.01, 0.7, 1.9, 1.2],  # id=6
            [0.65, 0.3, -1.6, 0.45],  # id=8
        ]
    )


@pytest.fixture
def index_id_map(services):
    return pd.DataFrame([2, 4, 6, 8], columns=["id"])


@pytest.fixture()
def weights():
    return torch.Tensor(
        [
            [0.2, -0.1, 2.0, 1.4],  # k=1
            [0.2, -0.1, 2.0, 1.4],  # k=2
        ]
    )


def test_proper_initialization(
    mongo, mocker, proper_parameters, services, service_embeddings, index_id_map
):
    mock_embedder_call = mocker.patch(
        "recommender.engines.autoencoders.ml_components.embedder.Embedder.__call__"
    )
    mock_embedder_call.return_value = (service_embeddings, index_id_map)

    FEATURES_DIM, _, SE = proper_parameters
    service_embedder = Embedder(AutoEncoder(FEATURES_DIM, SE))
    service_selector = ServiceSelector(service_embedder)

    assert service_selector.itemspace.shape == torch.Size([len(services), SE])
    assert service_selector.index_id_map.index.values.tolist() == list(
        range(len(services))
    )
    assert service_selector.index_id_map.id.values.tolist() == list([2, 4, 6, 8])
    mock_embedder_call.assert_called_once()


def test_call_with_matching_services(
    mongo,
    mocker,
    proper_parameters,
    services,
    service_embeddings,
    weights,
    index_id_map,
):
    mock_embedder_call = mocker.patch(
        "recommender.engines.autoencoders.ml_components.embedder.Embedder.__call__"
    )
    mock_embedder_call.return_value = (service_embeddings, index_id_map)

    FEATURES_DIM, K, SE = proper_parameters
    service_embedder = Embedder(AutoEncoder(FEATURES_DIM, SE))
    service_selector = ServiceSelector(service_embedder)

    assert service_selector(weights, mask=torch.ones(len(services)))[0] == [4, 6]
    assert service_selector(weights, mask=torch.Tensor([1, 1, 0, 1]))[0] == [4, 2]
    assert service_selector(weights, mask=torch.Tensor([1, 0, 0, 1]))[0] == [2, 8]
    assert service_selector(weights, mask=torch.Tensor([0, 0, 1, 1]))[0] == [6, 8]
    assert service_selector(weights, mask=torch.Tensor([0, 1, 0, 1]))[0] == [4, 8]

    scores = service_selector(weights, mask=torch.ones(len(services)))[1]
    assert len(scores) == K
    assert all(isinstance(s, float) for s in scores)
    assert scores == sorted(scores, reverse=True)


def test_raise_insufficient_recommendation_space(
    mongo,
    mocker,
    proper_parameters,
    services,
    weights,
    service_embeddings,
    index_id_map,
):
    mock_embedder_call = mocker.patch(
        "recommender.engines.autoencoders.ml_components.embedder.Embedder.__call__"
    )
    mock_embedder_call.return_value = (service_embeddings, index_id_map)

    FEATURES_DIM, K, SE = proper_parameters
    service_embedder = Embedder(AutoEncoder(FEATURES_DIM, SE))
    service_selector = ServiceSelector(service_embedder)

    with pytest.raises(InsufficientRecommendationSpaceError):
        service_selector(weights, mask=torch.zeros(len(services)))
        service_selector(weights, mask=torch.Tensor([0, 0, 0, 1]))
