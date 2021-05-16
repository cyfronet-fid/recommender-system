# pylint: disable-all

import pytest
import torch

from recommender.engine.agents.rl_agent.action_selector import ActionSelector
from recommender.engine.models.autoencoders import ServiceAutoEncoder, create_embedder
from recommender.errors import InsufficientRecommendationSpace
from recommender.models import SearchData, Service
from tests.factories.marketplace import ServiceFactory, UserFactory


@pytest.fixture
def proper_parameters():
    FEATURES_DIM = 10
    K = 2
    SE = 4
    I = 4
    return FEATURES_DIM, K, SE, I


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
def user(mongo):
    return UserFactory(accessed_services=[])


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


def test_proper_initialization(mocker, proper_parameters, services, service_embeddings):
    mock_torch_module_call = mocker.patch("torch.nn.Module.__call__")
    mock_forbidden_services = mocker.patch(
        "recommender.engine.agents.rl_agent.action_selector.retrieve_forbidden_services"
    )
    mock_forbidden_services.return_value = Service.objects(id__in=[2, 4])
    mock_torch_module_call.return_value = service_embeddings

    FEATURES_DIM, K, SE, I = proper_parameters
    service_embedder = create_embedder(ServiceAutoEncoder(FEATURES_DIM, SE))
    action_selector = ActionSelector(service_embedder)

    assert action_selector.itemspace_size == I
    assert action_selector.forbidden_services_size == 2
    assert action_selector.forbidden_indices == [0, 1]
    assert action_selector.index_id_map.index.values.tolist() == list(range(I))
    assert action_selector.index_id_map.id.values.tolist() == list([2, 4, 6, 8])
    mock_torch_module_call.assert_called_once()


def test_call_with_matching_services(
    mocker, proper_parameters, services, user, service_embeddings, weights
):
    mock_torch_module_call = mocker.patch("torch.nn.Module.__call__")
    mock_filter_services = mocker.patch(
        "recommender.engine.agents.rl_agent.action_selector.filter_services"
    )
    mock_forbidden_services = mocker.patch(
        "recommender.engine.agents.rl_agent.action_selector.retrieve_forbidden_services"
    )
    mock_torch_rand = mocker.patch("torch.rand")

    mock_torch_module_call.return_value = service_embeddings
    mock_forbidden_services.return_value = Service.objects(id__in=[-1])
    mock_torch_rand.return_value = torch.Tensor([0])

    FEATURES_DIM, K, SE, I = proper_parameters
    service_embedder = create_embedder(ServiceAutoEncoder(FEATURES_DIM, SE))
    action_selector = ActionSelector(service_embedder)

    mock_filter_services.return_value = Service.objects.all()
    assert action_selector(K, weights, user, search_data=SearchData()) == [4, 6]

    mock_filter_services.return_value = Service.objects(id__in=[2, 4, 8])
    assert action_selector(K, weights, user, search_data=SearchData()) == [4, 2]

    mock_filter_services.return_value = Service.objects(id__in=[-1])
    assert action_selector(K, weights, user, search_data=SearchData()) == [6, 8]


def test_call_with_forbidden_services(
    mocker, proper_parameters, services, user, service_embeddings, weights
):
    mock_torch_module_call = mocker.patch("torch.nn.Module.__call__")
    mock_filter_services = mocker.patch(
        "recommender.engine.agents.rl_agent.action_selector.filter_services"
    )
    mock_forbidden_services = mocker.patch(
        "recommender.engine.agents.rl_agent.action_selector.retrieve_forbidden_services"
    )
    mock_torch_rand = mocker.patch("torch.rand")

    mock_torch_module_call.return_value = service_embeddings
    mock_forbidden_services.return_value = Service.objects(id__in=[6])
    mock_torch_rand.return_value = torch.Tensor([0])

    FEATURES_DIM, K, SE, I = proper_parameters
    service_embedder = create_embedder(ServiceAutoEncoder(FEATURES_DIM, SE))
    action_selector = ActionSelector(service_embedder)

    mock_filter_services.return_value = Service.objects.all()
    assert action_selector(K, weights, user, search_data=SearchData()) == [4, 2]

    mock_filter_services.return_value = Service.objects(id__in=[2, 6, 8])
    assert action_selector(K, weights, user, search_data=SearchData()) == [2, 8]

    mock_filter_services.return_value = Service.objects(id__in=[-1])
    assert action_selector(K, weights, user, search_data=SearchData()) == [4, 8]


def test_raise_insufficient_recommendation_space(
    mocker, proper_parameters, services, weights, user, service_embeddings
):
    mock_torch_module_call = mocker.patch("torch.nn.Module.__call__")
    mock_forbidden_services = mocker.patch(
        "recommender.engine.agents.rl_agent.action_selector.retrieve_forbidden_services"
    )
    mock_forbidden_services.return_value = Service.objects.all()
    mock_torch_module_call.return_value = service_embeddings

    FEATURES_DIM, K, SE, I = proper_parameters
    service_embedder = create_embedder(ServiceAutoEncoder(FEATURES_DIM, SE))
    action_selector = ActionSelector(service_embedder)

    mock_forbidden_services.return_value = Service.objects.all()
    with pytest.raises(InsufficientRecommendationSpace):
        action_selector(K, weights, user, search_data=SearchData())
