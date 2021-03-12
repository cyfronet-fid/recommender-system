# pylint: disable-all

import torch
from torch.utils.data import DataLoader

from recommender.engine.pre_agent.datasets import PreAgentDataset
from recommender.engine.pre_agent.preprocessing import (
    USERS,
    SERVICES,
    precalc_users_and_service_tensors,
)
from tests.factories.populate_database import populate_users_and_services


def test_pre_agent_dataset(mongo):
    populate_users_and_services(
        common_services_number=4,
        no_one_services_number=1,
        users_number=4,
        k_common_services_min=1,
        k_common_services_max=3,
    )
    precalc_users_and_service_tensors()

    dataset = PreAgentDataset()
    train_ds_dl = DataLoader(dataset, batch_size=20)
    batch = next(iter(train_ds_dl))

    users_tensor = batch[USERS]
    assert isinstance(users_tensor, torch.Tensor)
    assert len(users_tensor) >= 8

    services_tensor = batch[SERVICES]
    assert isinstance(services_tensor, torch.Tensor)
    assert len(services_tensor) >= 8

    labels_tensor = batch[SERVICES]
    assert isinstance(labels_tensor, torch.Tensor)
    assert len(labels_tensor) >= 8
