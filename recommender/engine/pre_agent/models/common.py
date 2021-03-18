# pylint: disable=no-member, missing-class-docstring

"""This module contains common models functions"""

import io
import torch
from recommender.models import PytorchModule


def save_module(module, name=None, description=None):
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
