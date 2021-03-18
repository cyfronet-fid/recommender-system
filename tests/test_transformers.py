# pylint: disable-all

import pickle

import pytest
from sklearn.compose import ColumnTransformer

from recommender.engine.pre_agent.preprocessing import USERS, SERVICES, LABELS
from recommender.engine.pre_agent.preprocessing.transformers import (
    create_users_transformer,
    create_services_transformer,
    create_transformer,
    save_transformer,
    load_last_transformer, NoSavedTransformerError,
)
from recommender.models import ScikitLearnTransformer


def test_create_users_transformer(mongo):
    user_transformer = create_users_transformer()
    assert isinstance(user_transformer, ColumnTransformer)


def test_create_service_transformer(mongo):
    service_transformer = create_services_transformer()
    assert isinstance(service_transformer, ColumnTransformer)


def test_create_transformer(mongo):
    user_transformer = create_transformer(USERS)
    assert isinstance(user_transformer, ColumnTransformer)

    service_transformer = create_transformer(SERVICES)
    assert isinstance(service_transformer, ColumnTransformer)

    with pytest.raises(ValueError):
        create_transformer("placeholder_name")


def test_save_transformer(mongo):
    t = create_users_transformer()
    save_transformer(t, name="t", description="d")
    binary = ScikitLearnTransformer.objects(name="t").first().binary_transformer
    saved_t = pickle.loads(binary)
    assert isinstance(saved_t, ColumnTransformer)


def test_load_last_transformer(mongo):
    temp = create_users_transformer()
    save_transformer(temp, name="t", description="d")
    saved_t = load_last_transformer(name="t")
    assert isinstance(saved_t, ColumnTransformer)

    with pytest.raises(NoSavedTransformerError):
        load_last_transformer(name="placeholder_name")
