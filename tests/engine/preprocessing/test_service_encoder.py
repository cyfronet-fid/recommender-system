# pylint: disable-all

import pytest
import torch

from recommender.engines.rl.ml_components.service_encoder import (
    ServiceEncoder,
)
from tests.factories.search_data import SearchDataFactory
from tests.factories.state import StateFactory
from tests.factories.marketplace import ServiceFactory, UserFactory
from tests.factories.marketplace.category import CategoryFactory
from tests.factories.marketplace.scientific_domain import ScientificDomainFactory
from recommender.errors import SizeOfUsersAndElasticServicesError


@pytest.fixture
def categories(mongo):
    return CategoryFactory.create_batch(4)


@pytest.fixture
def scientific_domains(mongo):
    return ScientificDomainFactory.create_batch(4)


@pytest.fixture
def services(mongo, categories, scientific_domains):
    return [
        ServiceFactory(
            id=2, categories=categories[:2], scientific_domains=scientific_domains[2:]
        ),  # i=1
        ServiceFactory(
            id=8, categories=categories, scientific_domains=scientific_domains
        ),  # i=4
        ServiceFactory(
            id=4, categories=categories[2:], scientific_domains=[scientific_domains[3]]
        ),  # i=2
        ServiceFactory(
            id=6, categories=[categories[3]], scientific_domains=scientific_domains
        ),  # i=3
        ServiceFactory(id=11, status="draft"),  # i=5
        ServiceFactory(id=1, status="errored"),  # i=0
    ]


@pytest.fixture
def states(services):
    return [
        StateFactory(
            candidates=[services[0], services[1], services[-1]],
            search_data=SearchDataFactory(q=None),
            non_empty_history=True,
        ),
        StateFactory(
            candidates=[services[0], services[1], services[-1]],
            search_data=SearchDataFactory(q=None),
            non_empty_history=True,
        ),
    ]


@pytest.fixture
def users(mongo, services):
    return [
        UserFactory(accessed_services=[2, 1]),
        UserFactory(accessed_services=[6, 11]),
    ]


@pytest.fixture
def encoder(services):
    return ServiceEncoder()


def test_service_encoder_init(encoder, services, users):
    assert encoder.I == len(services)
    assert encoder.forbidden_service_indices == [0, 5]


def test_service_encoder_one_state_one_user(encoder):
    state = StateFactory(
        candidates=[2, 8, 4, 6, 11, 1],
        search_data=SearchDataFactory(q=None),
        non_empty_history=True,
    )
    user = UserFactory(accessed_services=[2, 1])

    desired_mask = torch.Tensor([[0, 0, 1, 1, 1, 0]])
    encoder_mask = encoder([user], [state])

    assert torch.equal(encoder_mask, desired_mask)


def test_service_encoder_multiple_states_users(encoder, states, users):
    desired_mask = torch.Tensor([[0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 1, 0]])

    assert torch.equal(encoder(users, states), desired_mask)


def test_service_encoder_exception(encoder):
    state = StateFactory(
        candidates=[2, 8, 4, 6, 11, 1],
        search_data=SearchDataFactory(q=None),
        non_empty_history=True,
    )
    user = [
        UserFactory(accessed_services=[2, 1]),
        UserFactory(accessed_services=[8, 2]),
    ]

    # 1 state and 2 users
    with pytest.raises(SizeOfUsersAndElasticServicesError):
        encoder(user, state)

    state = [
        StateFactory(
            candidates=[2, 8, 4, 6, 11, 1],
            search_data=SearchDataFactory(q=None),
            non_empty_history=True,
        ),
        StateFactory(
            candidates=[2, 8, 1],
            search_data=SearchDataFactory(q=None),
            non_empty_history=True,
        ),
    ]
    user = [UserFactory(accessed_services=[2, 1])]

    # 2 states and 1 user
    with pytest.raises(SizeOfUsersAndElasticServicesError):
        encoder(user, state)
