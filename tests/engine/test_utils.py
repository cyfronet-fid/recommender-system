# pylint: disable-all
import numpy as np
import pandas as pd
import pytest
import torch

from recommender.engine.models.autoencoders import ServiceAutoEncoder, create_embedder
from tests.factories.marketplace import ServiceFactory
from recommender.engine.agents.rl_agent.utils import (
    get_service_indices,
    embedded_tensors_exist,
    create_itemspace
)
from tests.factories.marketplace import UserFactory


@pytest.fixture
def proper_parameters():
    SOH = 10
    SE = 4
    return SOH, SE


@pytest.fixture
def services(mongo, proper_parameters):
    SOH, _ = proper_parameters
    return [
        ServiceFactory(id=2, tensor=torch.randint(2, (SOH,))),
        ServiceFactory(id=8, tensor=torch.randint(2, (SOH,))),
        ServiceFactory(id=4, tensor=torch.randint(2, (SOH,))),
        ServiceFactory(id=6, tensor=torch.randint(2, (SOH,))),
    ]


@pytest.fixture
def index_id_map():
    return pd.DataFrame([2, 4, 6, 8], columns=["id"])


def test_get_service_indices(index_id_map):
    assert get_service_indices(index_id_map, [4, 6]) == [1, 2]
    assert get_service_indices(index_id_map, [10]) == []
    assert get_service_indices(index_id_map, []) == []
    assert get_service_indices(index_id_map, [8, 2]) == [3, 0]


def test_create_itemspace(services, proper_parameters):
    SOH, SE = proper_parameters
    embedder = create_embedder(ServiceAutoEncoder(features_dim=SOH, embedding_dim=SE))
    itemspace, index_id_map = create_itemspace(embedder)
    assert itemspace.shape == torch.Size([len(services), SE])
    assert index_id_map.size == len(services)
    assert (index_id_map.id.values == np.array([2, 4, 6, 8])).all()
    assert (index_id_map.index.values == np.array([0, 1, 2, 3])).all()


def test_embedded_tensors_exist(mongo):
    UE = 32
    users = UserFactory.create_batch(3)
    assert embedded_tensors_exist(users) is False

    for user in users:
        user.embedded_tensor = torch.rand(UE).tolist()
        user.save()

    assert embedded_tensors_exist(users) is True

    users[1].embedded_tensor = []
    users[1].save()

    assert embedded_tensors_exist(users) is False
