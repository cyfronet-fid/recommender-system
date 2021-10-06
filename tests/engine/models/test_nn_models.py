# pylint: disable-all

import pytest
from torch.nn import Linear

from recommender.engine.utils import save_module, NoSavedModuleError, load_last_module


def test_save_and_load_last_module(mongo):
    with pytest.raises(NoSavedModuleError):
        load_last_module("placeholder_name")

    model = Linear(5, 5)

    save_module(model, name="placeholder_name")
    loaded_model = load_last_module("placeholder_name")

    assert isinstance(loaded_model, Linear)
    assert model.in_features == loaded_model.in_features
    assert model.out_features == loaded_model.out_features
