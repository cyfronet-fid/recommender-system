# pylint: disable-all

import pytest
import torch

from recommender.engine.agents.rl_agent.service_selector import ServiceSelector
from recommender.engine.models.autoencoders import ServiceAutoEncoder, create_embedder
from recommender.errors import InsufficientRecommendationSpace
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
        ServiceFactory(id=2, tensor=torch.randint(2, (FEATURES_DIM,))),
        ServiceFactory(id=8, tensor=torch.randint(2, (FEATURES_DIM,))),
        ServiceFactory(id=4, tensor=torch.randint(2, (FEATURES_DIM,))),
        ServiceFactory(id=6, tensor=torch.randint(2, (FEATURES_DIM,))),
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


@pytest.fixture()
def weights():
    return torch.Tensor(
        [
            [0.2, -0.1, 2.0, 1.4],  # k=1
            [0.2, -0.1, 2.0, 1.4],  # k=2
        ]
    )


def test_proper_initialization(
    mongo, mocker, proper_parameters, services, service_embeddings
):
    mock_torch_module_call = mocker.patch("torch.nn.Module.__call__")
    mock_torch_module_call.return_value = service_embeddings

    FEATURES_DIM, _, SE = proper_parameters
    service_embedder = create_embedder(ServiceAutoEncoder(FEATURES_DIM, SE))
    service_selector = ServiceSelector(service_embedder)

    assert service_selector.itemspace_size == len(services)
    assert service_selector.index_id_map.index.values.tolist() == list(
        range(len(services))
    )
    assert service_selector.index_id_map.id.values.tolist() == list([2, 4, 6, 8])
    mock_torch_module_call.assert_called_once()


def test_call_with_matching_services(
    mongo, mocker, proper_parameters, services, service_embeddings, weights
):
    mock_torch_module_call = mocker.patch("torch.nn.Module.__call__")
    mock_torch_module_call.return_value = service_embeddings

    FEATURES_DIM, K, SE = proper_parameters
    service_embedder = create_embedder(ServiceAutoEncoder(FEATURES_DIM, SE))
    service_selector = ServiceSelector(service_embedder)

    assert service_selector(K, weights, mask=torch.ones(len(services))) == [4, 6]
    assert service_selector(K, weights, mask=torch.Tensor([1, 1, 0, 1])) == [4, 2]
    assert service_selector(K, weights, mask=torch.Tensor([1, 0, 0, 1])) == [2, 8]
    assert service_selector(K, weights, mask=torch.Tensor([0, 0, 1, 1])) == [6, 8]
    assert service_selector(K, weights, mask=torch.Tensor([0, 1, 0, 1])) == [4, 8]


def test_raise_insufficient_recommendation_space(
    mongo, mocker, proper_parameters, services, weights, service_embeddings
):
    mock_torch_module_call = mocker.patch("torch.nn.Module.__call__")
    mock_torch_module_call.return_value = service_embeddings

    FEATURES_DIM, K, SE = proper_parameters
    service_embedder = create_embedder(ServiceAutoEncoder(FEATURES_DIM, SE))
    service_selector = ServiceSelector(service_embedder)

    with pytest.raises(InsufficientRecommendationSpace):
        service_selector(K, weights, mask=torch.zeros(len(services)))
        service_selector(K, weights, mask=torch.Tensor([0, 0, 0, 1]))
