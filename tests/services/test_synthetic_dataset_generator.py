# pylint: disable-all
import pandas as pd
import pytest
import torch
import numpy as np

from recommender.engine.agents.rl_agent.reward_mapping import (
    TRANSITION_REWARDS_CSV_PATH,
)
from recommender.services.synthetic_dataset_generator import (
    _normalize_embedded_services,
    get_service_indices,
    iou,
    approx_service_engagement,
    construct_reward,
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
            [0.1253, 0.0696, 0.2228, -0.2785],  # id=2
            [0.1114, -0.2089, 0.4178, 0.1393],  # id=4
            [-0.0014, 0.0975, 0.2646, 0.1671],  # id=6
            [0.0905, 0.0418, -0.2228, 0.0627],  # id=8
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


def test__normalize_embedded_services(service_embeddings, normalized_embedded_services):
    assert torch.allclose(
        _normalize_embedded_services(service_embeddings),
        normalized_embedded_services,
        atol=10e-5,
    )


def test_approx_service_engagement(
    user, services, normalized_embedded_services, index_id_map
):
    calculated_s6 = np.array(
        approx_service_engagement(
            user, services[2], normalized_embedded_services, index_id_map
        )
    )
    desired_s6 = 0.5605
    assert np.isclose(calculated_s6, desired_s6, atol=1e-04)

    calculated_s8 = np.array(
        approx_service_engagement(
            user, services[3], normalized_embedded_services, index_id_map
        )
    )
    desired_s8 = 0.4764
    assert np.isclose(calculated_s8, desired_s8, atol=1e-04)


def test_construct_rewards():
    transitions_df = pd.read_csv(TRANSITION_REWARDS_CSV_PATH, index_col="source")
    repetitions = 1000

    high_engagement_buffer = []

    for _ in range(repetitions):
        high_engagement_buffer.append(construct_reward(transitions_df, 1.0))

    he_orders_percent = (
        len(
            list(
                filter(
                    lambda x: len(x) > 0 and x[-1] == "order", high_engagement_buffer
                )
            )
        )
        / repetitions
    )
    he_empty_percent = (
        len(list(filter(lambda x: len(x) == 0, high_engagement_buffer))) / repetitions
    )

    assert he_orders_percent > 0.8
    assert he_empty_percent < 0.2

    low_engagement_buffer = []

    for _ in range(repetitions):
        low_engagement_buffer.append(construct_reward(transitions_df, 0.0))

    le_orders_percent = (
        len(
            list(
                filter(lambda x: len(x) > 0 and x[-1] == "order", low_engagement_buffer)
            )
        )
        / repetitions
    )
    le_empty_percent = (
        len(list(filter(lambda x: len(x) == 0, low_engagement_buffer))) / repetitions
    )

    assert le_orders_percent < 0.2
    assert le_empty_percent > 0.8
