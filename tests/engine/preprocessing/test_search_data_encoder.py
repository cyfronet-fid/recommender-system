# pylint: disable-all

import pytest
import torch

from recommender.engine.agents.rl_agent.preprocessing.search_data_encoder import (
    SearchDataEncoder,
)
from recommender.models import SearchData
from tests.factories.marketplace import ServiceFactory, UserFactory
from tests.factories.marketplace.category import CategoryFactory
from tests.factories.marketplace.scientific_domain import ScientificDomainFactory


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
def users(mongo, services):
    return [
        UserFactory(accessed_services=[2, 1]),
        UserFactory(accessed_services=[6, 11]),
    ]


@pytest.fixture
def encoder(mongo, services):
    return SearchDataEncoder()


def test_search_data_encoder_init(encoder, services, users):
    assert encoder.itemspace_size == len(services)
    assert encoder.forbidden_service_indices == [0, 5]


def test_search_data_encoder_empty_search(encoder, services, users):
    desired_mask = torch.Tensor([[0, 0, 1, 1, 1, 0], [0, 1, 1, 0, 1, 0]])

    encoder_mask = encoder(users, [SearchData()] * len(users))
    assert torch.equal(encoder_mask, desired_mask)


def test_search_data_encoder_full_search(
    encoder, services, users, categories, scientific_domains
):
    search_data = SearchData(
        categories=categories, scientific_domains=scientific_domains
    )

    desired_mask = torch.Tensor([[0, 0, 1, 1, 1, 0], [0, 1, 1, 0, 1, 0]])

    encoder_mask = encoder(users, [search_data] * len(users))
    assert torch.equal(encoder_mask, desired_mask)


def test_search_data_encoder_single_search(
    encoder, services, users, categories, scientific_domains
):
    sd1 = SearchData(categories=categories[:2])
    sd2 = SearchData(categories=categories[2:])
    sd3 = SearchData(scientific_domains=scientific_domains[:2])
    sd4 = SearchData(scientific_domains=[scientific_domains[2]])

    desired_mask1 = torch.Tensor(
        [
            [0, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 1, 0],
        ]
    )

    desired_mask2 = torch.Tensor([[0, 0, 0, 1, 1, 0], [0, 1, 0, 0, 1, 0]])

    assert torch.equal(encoder(users, [sd1, sd2]), desired_mask1)
    assert torch.equal(encoder(users, [sd3, sd4]), desired_mask2)


def test_search_data_encoder_multiple_search(
    encoder, services, users, categories, scientific_domains
):
    sd1 = SearchData(
        categories=[categories[1]], scientific_domains=[scientific_domains[2]]
    )
    sd2 = SearchData(
        categories=[categories[0]], scientific_domains=[scientific_domains[3]]
    )

    desired_mask = torch.Tensor([[0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 1, 0]])

    encoder_mask = encoder(users, [sd1, sd2])
    assert torch.equal(encoder_mask, desired_mask)
