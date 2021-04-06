# pylint: disable=no-member, missing-class-docstring

"""This module contains common datasets functions"""


import io
import torch

from recommender.models import PytorchDataset

TRAIN = "training"
VALID = "validation"
TEST = "testing"


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
