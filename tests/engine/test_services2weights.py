# pylint: disable-all
import pandas as pd
import pytest
import torch

from recommender.engines.rl.ml_components.service_selector import ServiceSelector
from recommender.engines.rl.ml_components.services2weights import Services2Weights
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


@pytest.fixture
def index_id_map(services):
    return pd.DataFrame([2, 4, 6, 8], columns=["id"])


@pytest.mark.skip(reason="TODO")
def test_services2weights(mocker, services, service_embeddings, index_id_map):
    # Values of SE and Service OH len don't matter here
    # as we are mocking the service embedder output anyway
    mock_embedder_call = mocker.patch(
        "recommender.engines.autoencoders.ml_components.embedder.Embedder.__call__"
    )
    mock_embedder_call.return_value = (service_embeddings, index_id_map)

    services2weights = Services2Weights()
    service_selector = ServiceSelector()

    recommended_ids_list_v1 = [[2, 4, 8], [4, 8, 2], [2, 6, 8], [2, 4, 6], [6, 8, 2]]
    recommended_ids_list_v2 = [[2, 4], [4, 8], [2, 6], [2, 4], [6, 8]]

    for recommended_ids_list in (recommended_ids_list_v1, recommended_ids_list_v2):
        weights_batch = services2weights(recommended_ids_list)
        for recommended_ids, weights in zip(recommended_ids_list, weights_batch):
            assert (
                service_selector(
                    weights=weights,
                    mask=torch.ones(len(services)),
                )
                == recommended_ids
            )
