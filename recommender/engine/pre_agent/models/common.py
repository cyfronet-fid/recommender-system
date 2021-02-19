# pylint: disable=no-member

"""This module contains common models functions"""

import pickle
from recommender.models import PytorchModule


def save_module(module, name=None, description=None):
    """It saves model to database using pickle"""
    PytorchModule(
        name=name, description=description, binary_module=pickle.dumps(module)
    ).save()


def load_last_module(name):
    """It loads model from database and unpickles it"""
    module = pickle.loads(
        PytorchModule.objects(name=name).order_by("-id").first().binary_module
    )
    module.eval()

    return module
