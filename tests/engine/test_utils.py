# pylint: disable-all

import pandas as pd
import pytest
import torch

from recommender.engine.agents.rl_agent.utils import get_service_indices, iou


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


def test_get_service_indices(service_embeddings, index_id_map):
    assert get_service_indices(index_id_map, [4, 6]) == [1, 2]
    assert get_service_indices(index_id_map, [10]) == []
    assert get_service_indices(index_id_map, []) == []
    assert get_service_indices(index_id_map, [8, 2]) == [3, 0]


def test_iou():
    s1 = {0, 1, 2, 3}
    s2 = {2, 3, 4, 5}
    assert iou(s1, s2) == 1 / 3
