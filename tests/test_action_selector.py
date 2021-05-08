# pylint: disable-all

import pytest
import torch

from recommender.engine.pre_agent.models import ServiceAutoEncoder
from recommender.engine.rl_agent.action_selector import ActionSelector
from recommender.errors import InsufficientRecommendationSpace
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
            [0.2, -0.1, 2.0, 1.4],
            [0.2, -0.1, 2.0, 1.4],
        ]
    )


def test_proper_initialization(mocker, proper_parameters, services):
    mock_torch_module_call = mocker.patch("torch.nn.Module.__call__")
    mock_forbidden_ids = mocker.patch(
        "recommender.engine.rl_agent.action_selector.retrieve_forbidden_service_ids"
    )
    mock_forbidden_ids.return_value = [2, 4]

    FEATURES_DIM, K, SE, I = proper_parameters
    action_selector = ActionSelector(K, ServiceAutoEncoder(FEATURES_DIM, SE).encoder)

    assert action_selector.K == K
    assert action_selector.I == I
    assert action_selector.index_id_map.index.values.tolist() == list(range(I))
    assert action_selector.index_id_map.id.values.tolist() == list([2, 4, 6, 8])
    mock_torch_module_call.assert_called_once()

    mock_forbidden_ids.return_value = [2, 4, 6]
    with pytest.raises(InsufficientRecommendationSpace):
        ActionSelector(K, ServiceAutoEncoder(FEATURES_DIM, SE).encoder)


def test_call_with_matching_services(
    mocker, proper_parameters, services, user, service_embeddings, weights
):
    mock_torch_module_call = mocker.patch("torch.nn.Module.__call__")
    mock_retrieve_service_ids = mocker.patch(
        "recommender.engine.rl_agent.action_selector.retrieve_matching_service_ids"
    )
    mock_forbidden_ids = mocker.patch(
        "recommender.engine.rl_agent.action_selector.retrieve_forbidden_service_ids"
    )
    mock_torch_rand = mocker.patch("torch.rand")

    mock_torch_module_call.return_value = service_embeddings
    mock_forbidden_ids.return_value = []
    mock_torch_rand.return_value = torch.Tensor([0])

    FEATURES_DIM, K, SE, I = proper_parameters
    action_selector = ActionSelector(K, ServiceAutoEncoder(FEATURES_DIM, SE).encoder)

    mock_retrieve_service_ids.return_value = [2, 4, 6, 8]
    assert action_selector(weights, user, search_data={}) == [4, 6]

    mock_retrieve_service_ids.return_value = [2, 4, 8]
    assert action_selector(weights, user, search_data={}) == [4, 2]

    mock_retrieve_service_ids.return_value = []
    assert action_selector(weights, user, search_data={}) == [6, 8]


def test_call_with_forbidden_services(
    mocker, proper_parameters, services, user, service_embeddings, weights
):
    mock_torch_module_call = mocker.patch("torch.nn.Module.__call__")
    mock_retrieve_service_ids = mocker.patch(
        "recommender.engine.rl_agent.action_selector.retrieve_matching_service_ids"
    )
    mock_forbidden_ids = mocker.patch(
        "recommender.engine.rl_agent.action_selector.retrieve_forbidden_service_ids"
    )
    mock_torch_rand = mocker.patch("torch.rand")

    mock_torch_module_call.return_value = service_embeddings
    mock_forbidden_ids.return_value = [6]
    mock_torch_rand.return_value = torch.Tensor([0])

    FEATURES_DIM, K, SE, I = proper_parameters
    action_selector = ActionSelector(K, ServiceAutoEncoder(FEATURES_DIM, SE).encoder)

    mock_retrieve_service_ids.return_value = [2, 4, 6, 8]
    assert action_selector(weights, user, search_data={}) == [4, 2]

    mock_retrieve_service_ids.return_value = [2, 6, 8]
    assert action_selector(weights, user, search_data={}) == [2, 8]

    mock_retrieve_service_ids.return_value = []
    assert action_selector(weights, user, search_data={}) == [4, 8]
