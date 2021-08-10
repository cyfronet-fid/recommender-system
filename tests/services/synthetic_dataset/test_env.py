# pylint: disable-all

from time import time

import pandas as pd
import pytest
import torch

from recommender.services.synthetic_dataset.env import SyntheticMP, NoSyntheticUsers
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
def users(categories, scientific_domains):
    return [
        UserFactory(
            categories=categories,
            scientific_domains=scientific_domains,
            accessed_services=[],
            synthetic=True,
        ),
        UserFactory(
            categories=[], scientific_domains=[], accessed_services=[], synthetic=True
        ),
    ]


@pytest.fixture
def interactions_per_user():
    return 100


@pytest.fixture
def env(
    mocker, users, services, service_embeddings, index_id_map, interactions_per_user
):
    load_last_module_mock = mocker.patch(
        "recommender.services.synthetic_dataset.env.load_last_module"
    )
    load_last_module_mock.return_value = None

    create_embedder_mock = mocker.patch(
        "recommender.services.synthetic_dataset.env.create_embedder"
    )
    create_embedder_mock.return_value = None

    use_service_embedder_mock = mocker.patch(
        "recommender.services.synthetic_dataset.env.use_service_embedder"
    )
    use_service_embedder_mock.return_value = (service_embeddings, index_id_map)

    return SyntheticMP(
        interactions_per_user=interactions_per_user, advanced_search_data=False
    )


def test_env_init(mongo):
    with pytest.raises(NoSyntheticUsers):
        SyntheticMP(advanced_search_data=False)


def test_env_reset(env, users):
    assert env.current_user is None
    assert env.engaged_services is None
    assert env.ordered_services is None
    assert env.interaction is None

    state = env.reset()

    assert env.current_user in users
    assert env.engaged_services == []
    assert env.ordered_services == []
    assert env.interaction == 0
    assert state.user == env.current_user
    assert state.services_history == []
    assert state.synthetic

    prev_user = env.current_user
    env.reset()

    assert env.current_user != prev_user


def test_env_step(env, users, services):
    env.reset()
    action = services[:3]

    if env.current_user == users[0]:
        state, reward, done = env.step(action)
        assert state.user == users[0]
        assert state.services_history == action
        assert all(reward)
        assert not done
    else:
        action = services[:3]
        state, reward, done = env.step(action)
        assert state.user == users[1]
        assert state.services_history == []
        assert not all(reward)
        assert not done


def test_env_interactions(env, services, interactions_per_user):
    env.reset()
    action = services[:3]

    done = False
    interactions = -1
    start = time()

    while not done:
        interactions += 1
        state, reward, done = env.step(action)

    end = time()

    assert interactions == interactions_per_user
    assert end - start < interactions_per_user * 0.2
