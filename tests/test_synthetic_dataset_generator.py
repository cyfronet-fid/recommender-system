# pylint: disable-all
import pandas as pd
import pytest
import torch
import numpy as np

from recommender.services.synthetic_dataset_generator import (
    _normalize_embedded_services,
    get_service_indices,
    iou,
    predict_user_interest,
)
from tests.factories.marketplace import ServiceFactory, UserFactory
from tests.factories.marketplace.category import CategoryFactory
from tests.factories.marketplace.scientific_domain import ScientificDomainFactory


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
            [0.1253, 0.0696, 0.2228, -0.2785],
            [0.1114, -0.2089, 0.4178, 0.1393],
            [-0.0014, 0.0975, 0.2646, 0.1671],
            [0.0905, 0.0418, -0.2228, 0.0627],
        ]
    )


@pytest.fixture
def index_id_map():
    return pd.DataFrame([2, 4, 6, 8], columns=["id"])


@pytest.fixture
def categories(mongo):
    return CategoryFactory.create_batch(4)


@pytest.fixture
def scientific_domains(mongo):
    return ScientificDomainFactory.create_batch(4)


@pytest.fixture
def services(categories, scientific_domains):
    return [
        ServiceFactory(
            id=2, categories=categories, scientific_domains=scientific_domains
        ),
        ServiceFactory(
            id=4, categories=categories, scientific_domains=scientific_domains
        ),
        ServiceFactory(
            id=6, categories=categories, scientific_domains=scientific_domains
        ),
        ServiceFactory(
            id=8, categories=categories, scientific_domains=scientific_domains
        ),
    ]


@pytest.fixture
def user(services, categories, scientific_domains):
    return UserFactory(
        categories=categories[:2],
        scientific_domains=scientific_domains[:2],
        accessed_services=services[:2],
    )


def test_normalize_embedded_services(service_embeddings, normalized_embedded_services):
    assert torch.allclose(
        _normalize_embedded_services(service_embeddings),
        normalized_embedded_services,
        atol=10e-5,
    )


def test_get_service_indices(service_embeddings, index_id_map):
    assert get_service_indices(index_id_map, [4, 6]) == [1, 2]
    assert get_service_indices(index_id_map, [10]) == []
    assert get_service_indices(index_id_map, []) == []
    assert get_service_indices(index_id_map, [8, 2]) == [0, 3]


def test_iou():
    s1 = {0, 1, 2, 3}
    s2 = {2, 3, 4, 5}
    assert iou(s1, s2) == 1 / 3


def test_predict_user_interest(
    user, services, normalized_embedded_services, index_id_map
):
    calculated = np.array(
        predict_user_interest(
            user, services[2:], normalized_embedded_services, index_id_map
        )
    )
    desired = np.array([0.4395, 0.5236])
    assert np.allclose(calculated, desired, 10e-3)
