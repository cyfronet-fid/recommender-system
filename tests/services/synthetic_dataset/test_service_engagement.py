# pylint: disable-all

import numpy as np
import pandas as pd
import torch
import pytest

from recommender.services.synthetic_dataset.service_engagement import (
    approx_service_engagement,
)
from tests.factories.marketplace import UserFactory, ServiceFactory
from tests.factories.marketplace.category import CategoryFactory
from tests.factories.marketplace.scientific_domain import ScientificDomainFactory


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


def test_approx_service_engagement(
    user, services, normalized_embedded_services, index_id_map
):
    calculated_s6 = np.array(
        approx_service_engagement(
            user, services[2], services[:2], normalized_embedded_services, index_id_map
        )
    )
    desired_s6 = 0.5605
    assert np.isclose(calculated_s6, desired_s6, atol=1e-04)

    calculated_s8 = np.array(
        approx_service_engagement(
            user, services[3], services[:2], normalized_embedded_services, index_id_map
        )
    )
    desired_s8 = 0.4764
    assert np.isclose(calculated_s8, desired_s8, atol=1e-04)