# pylint: disable=missing-function-docstring, no-member, not-callable

"""This module contains function that creates datasets needed for autoencoders
 training"""

import torch
from torch.utils.data import random_split

from recommender.engine.pre_agent.datasets.common import TRAIN, VALID, TEST
from recommender.engine.pre_agent.preprocessing import USERS, SERVICES
from recommender.models import User, Service


AUTOENCODERS = "autoencoders"


def get_autoencoder_dataset_name(collection_name, split):
    """Get autoencoder dataset name"""

    valid_colection_names = (USERS, SERVICES)
    if collection_name not in valid_colection_names:
        raise ValueError(
            f"Invalid collection_name, should be one of: {valid_colection_names}"
        )

    valid_splits = (TRAIN, VALID, TEST)
    if split not in valid_splits:
        raise ValueError(f"Invalid split, should be one of: {valid_splits}")

    return f"{AUTOENCODERS} {collection_name} {split} dataset"


def create_autoencoder_datasets(
    collection_name, train_ds_size=0.6, valid_ds_size=0.2, device=torch.device("cpu")
):
    """Creates train/valid/test datasets for users/services autoencoder"""

    if collection_name == USERS:
        collection = list(User.objects)
    elif collection_name == SERVICES:
        collection = list(Service.objects)
    else:
        raise ValueError

    ds_tensor = torch.stack([torch.tensor(obj.tensor) for obj in collection])
    ds_tensor = ds_tensor.to(device)
    dataset = torch.utils.data.TensorDataset(ds_tensor)

    ds_size = len(dataset)
    train_ds_size = int(train_ds_size * ds_size)
    valid_ds_size = int(valid_ds_size * ds_size)
    test_ds_size = ds_size - (train_ds_size + valid_ds_size)

    train_ds, valid_ds, test_ds = random_split(
        dataset, [train_ds_size, valid_ds_size, test_ds_size]
    )

    output = {TRAIN: train_ds, VALID: valid_ds, TEST: test_ds}

    return output
