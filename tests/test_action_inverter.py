# pylint: disable-all

import pytest
import torch

from recommender.engine.agents.rl_agent.action_inverter import ActionInverter
from recommender.engine.agents.rl_agent.action_selector import ActionSelector
from recommender.engine.models.autoencoders import ServiceAutoEncoder, create_embedder
from tests.factories.marketplace import ServiceFactory


@pytest.fixture
def services(mongo):
    return [
        ServiceFactory(id=2),
        ServiceFactory(id=4),
        ServiceFactory(id=6),
        ServiceFactory(id=8),
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


def test_action_inverter(mocker, services, service_embeddings):
    # Values of SE and Service OH len don't matter here
    # as we are mocking the service embedder output anyway
    service_embedder = create_embedder(ServiceAutoEncoder(64, 4))
    mock_torch_module_call = mocker.patch("torch.nn.Module.__call__")
    mock_torch_module_call.return_value = service_embeddings

    action_inverter = ActionInverter(service_embedder)
    action_selector = ActionSelector(service_embedder)

    recommended_ids_list_v1 = [[2, 4, 8], [4, 8, 2], [2, 6, 8], [2, 4, 6], [6, 8, 2]]
    recommended_ids_list_v2 = [[2, 4], [4, 8], [2, 6], [2, 4], [6, 8]]

    for recommended_ids_list in (recommended_ids_list_v1, recommended_ids_list_v2):
        weights_batch = action_inverter(recommended_ids_list)
        for recommended_ids, weights in zip(recommended_ids_list, weights_batch):
            assert (
                action_selector(
                    K=len(recommended_ids),
                    weights=weights,
                    mask=torch.ones(len(services)),
                )
                == recommended_ids
            )
