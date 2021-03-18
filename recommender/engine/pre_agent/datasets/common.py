# pylint: disable=no-member, missing-class-docstring

"""This module contains common datasets functions"""
import io

import torch
from torch.utils.data import random_split

from recommender.engine.pre_agent.datasets.pre_agent_dataset import PreAgentDataset
from recommender.models import PytorchDataset

TRAIN_DS_SIZE = 0.6
VALID_DS_SIZE = 0.2
TEST_DS_SIZE = 1 - (TRAIN_DS_SIZE + VALID_DS_SIZE)

TRAIN_DS_NAME = "training dataset"
VALID_DS_NAME = "validation dataset"
TEST_DS_NAME = "testing dataset"


def save_dataset(dataset, name=None, description=None):
    """It saves model to the database"""
    buffer = io.BytesIO()
    torch.save(dataset, buffer)
    dataset_bytes = buffer.getvalue()

    PytorchDataset(
        name=name, description=description, dataset_bytes=dataset_bytes
    ).save()


class NoSavedDatasetError(Exception):
    pass


def load_last_dataset(name):
    """It loads model from the database"""

    last_dataset_model = PytorchDataset.objects(name=name).order_by("-id").first()

    if last_dataset_model is None:
        raise NoSavedDatasetError(f"No saved dataset with name {name}!")

    dataset_bytes = last_dataset_model.dataset_bytes
    buffer = io.BytesIO(dataset_bytes)
    dataset = torch.load(buffer)

    return dataset


def create_datasets(
    train_ds_size=TRAIN_DS_SIZE,
    valid_ds_size=VALID_DS_SIZE,
    train_ds_name=TRAIN_DS_NAME,
    valid_ds_name=VALID_DS_NAME,
    test_ds_name=TEST_DS_NAME,
):
    """Creates Pre Agent Dataset, split it into train/valid/test datasets and
    saves each of them."""

    dataset = PreAgentDataset()

    ds_size = len(dataset)
    train_ds_size = int(train_ds_size * ds_size)
    valid_ds_size = int(valid_ds_size * ds_size)
    test_ds_size = ds_size - (train_ds_size + valid_ds_size)

    print("Spliting into Train/Valid/Test datasets...")
    train_ds, valid_ds, test_ds = random_split(
        dataset, [train_ds_size, valid_ds_size, test_ds_size]
    )
    print("finished!")

    print("Train/Valid/Test datasets saving...")
    save_dataset(train_ds, name=train_ds_name)
    save_dataset(valid_ds, name=valid_ds_name)
    save_dataset(test_ds, name=test_ds_name)
    print("finished!")

    return train_ds, valid_ds, test_ds
