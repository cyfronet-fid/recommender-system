# pylint: disable-all

import pytest

from engine.pre_agent.models import NeuralColaborativeFilteringModel
from engine.pre_agent.models.common import NoSavedModuleError, load_last_module, save_module


def test_save_and_load_last_module(mongo):
    with pytest.raises(NoSavedModuleError):
        load_last_module("placeholder_name")

    model = NeuralColaborativeFilteringModel(1, 2, 3, 4)
    save_module(model, name="placeholder_name")
    loaded_model = load_last_module("placeholder_name")

    assert isinstance(loaded_model, NeuralColaborativeFilteringModel)
    assert model.user_features_dim == loaded_model.user_features_dim
    assert model.user_embedding_dim == loaded_model.user_embedding_dim
    assert model.service_features_dim == loaded_model.service_features_dim
    assert model.service_embedding_dim == loaded_model.service_embedding_dim
