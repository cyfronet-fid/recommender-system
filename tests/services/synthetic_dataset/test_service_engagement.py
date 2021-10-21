# pylint: disable-all

import numpy as np
import pandas as pd
import torch
import pytest

from recommender.services.synthetic_dataset.service_engagement import (
    approx_service_engagement,
    _distance_metric,
    _compute_distance_score,
    _compute_overlap_score,
)
from tests.factories.marketplace import UserFactory, ServiceFactory
from tests.factories.marketplace.category import CategoryFactory
from tests.factories.marketplace.scientific_domain import ScientificDomainFactory


@pytest.fixture
def normalized_embedded_services():
    return torch.Tensor(
        [
            [0.1672, 0.1180, 0.5648],
            [-0.4665, -0.2208, 0.0748],
            [0.2713, -0.4302, -0.1070],
            [0.4049, -0.8003, -0.4423],
        ]
    )


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
    )


@pytest.fixture
def index_id_map():
    return pd.DataFrame([2, 4, 6, 8], columns=["id"])


def test__distance_measure(normalized_embedded_services):
    cluster = normalized_embedded_services[1:]
    point = normalized_embedded_services[0]
    calculated_distance = _distance_metric(cluster, point)
    desired_distance = 0.5272
    assert np.isclose(calculated_distance, desired_distance, atol=1e-4)


def test__compute_distance_score(services, normalized_embedded_services, index_id_map):
    calculated_score = _compute_distance_score(
        services[:-1], services[-1], normalized_embedded_services, index_id_map
    )
    desired_score = 0.5143
    assert np.isclose(calculated_score, desired_score, atol=1e-4)


def test__compute_overlap_score(user, services):
    calculated_score = _compute_overlap_score(user, services[0])
    desired_score = 0.9525
    assert np.isclose(calculated_score, desired_score, atol=1e-4)


def test_approx_service_engagement(
    user, services, normalized_embedded_services, index_id_map
):
    calculated_s6 = np.array(
        approx_service_engagement(
            user, services[0], services[1:], normalized_embedded_services, index_id_map
        )
    )
    desired_s6 = 0.7399
    assert np.isclose(calculated_s6, desired_s6, atol=1e-4)

    calculated_s8 = np.array(
        approx_service_engagement(
            user,
            services[-1],
            services[:-1],
            normalized_embedded_services,
            index_id_map,
        )
    )
    desired_s8 = 0.7334
    assert np.isclose(calculated_s8, desired_s8, atol=1e-4)
