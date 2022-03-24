# pylint: disable-all

import torch
import pytest

from recommender.engines.rl.ml_components.synthetic_dataset.dataset import (
    _normalize_embedded_services,
)


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
def normalized_embedded_services():
    return torch.Tensor(
        [
            [0.1253, 0.0696, 0.2228, -0.2785],  # id=2
            [0.1114, -0.2089, 0.4178, 0.1393],  # id=4
            [-0.0014, 0.0975, 0.2646, 0.1671],  # id=6
            [0.0905, 0.0418, -0.2228, 0.0627],  # id=8
        ]
    )


def test__normalize_embedded_services(service_embeddings, normalized_embedded_services):
    assert torch.allclose(
        _normalize_embedded_services(service_embeddings),
        normalized_embedded_services,
        atol=10e-5,
    )
