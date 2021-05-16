# pylint: disable=missing-class-docstring, no-member

"""Recommender engine utilities"""

import io
from typing import Optional

import torch

from recommender.models import PytorchModule, PytorchDataset


def save_module(
    module, name: Optional[str] = None, description: Optional[str] = None
) -> None:
    """It saves module to the database"""

    buffer = io.BytesIO()
    torch.save(module, buffer)
    module_bytes = buffer.getvalue()

    PytorchModule(name=name, description=description, module_bytes=module_bytes).save()


class NoSavedModuleError(Exception):
    pass


def load_last_module(name):
    """It loads module from the database"""
    last_module_model = PytorchModule.objects(name=name).order_by("-id").first()

    if last_module_model is None:
        raise NoSavedModuleError(f"No saved module (model) with name {name}!")

    module_bytes = last_module_model.module_bytes
    buffer = io.BytesIO(module_bytes)
    module = torch.load(buffer)
    module.eval()

    return module


TRAIN = "training"
VALID = "validation"
TEST = "testing"


def save_dataset(
    dataset, name: Optional[str] = None, description: Optional[str] = None
) -> None:
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
