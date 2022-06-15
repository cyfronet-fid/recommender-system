# pylint: disable-all
import pandas as pd
import pytest
import torch

from tests.factories.marketplace import ServiceFactory
from recommender.engines.rl.utils import (
    get_service_indices,
)


@pytest.fixture
def proper_parameters():
    SOH = 10
    SE = 4
    return SOH, SE


@pytest.fixture
def services(mongo, proper_parameters):
    SOH, _ = proper_parameters
    return [
        ServiceFactory(id=2),
        ServiceFactory(id=8),
        ServiceFactory(id=4),
        ServiceFactory(id=6),
    ]


@pytest.fixture
def index_id_map():
    return pd.DataFrame([2, 4, 6, 8], columns=["id"])


def test_get_service_indices(index_id_map):
    assert get_service_indices(index_id_map, [4, 6]) == [1, 2]
    assert get_service_indices(index_id_map, [10]) == []
    assert get_service_indices(index_id_map, []) == []
    assert get_service_indices(index_id_map, [8, 2]) == [3, 0]
