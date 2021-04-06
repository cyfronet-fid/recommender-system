# pylint: disable-all

import pytest
import torch
from torch.utils.data import DataLoader

from recommender.engine.pre_agent.datasets.all_datasets import create_datasets
from recommender.engine.pre_agent.datasets.common import (
    NoSavedDatasetError,
    load_last_dataset,
)
from recommender.engine.pre_agent.preprocessing import (
    USERS,
    SERVICES,
    precalc_users_and_service_tensors,
)
from tests.factories.populate_database import populate_users_and_services


def test_load_last_dataset(mongo):
    with pytest.raises(NoSavedDatasetError):
        load_last_dataset("placeholder_name")
