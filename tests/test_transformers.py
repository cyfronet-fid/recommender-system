# pylint: disable-all

import pickle

from sklearn.compose import ColumnTransformer

from recommender.engine.pre_agent.preprocessing.mongo_to_dataframe import (
    USERS,
    SERVICES,
    LABELS,
)
from recommender.engine.pre_agent.preprocessing.transformers import (
    create_users_transformer,
    create_services_transformer,
    create_labels_transformer,
    create_transformer,
    create_transformers,
    save_transformer,
    load_last_transformer,
)
from recommender.models import ScikitLearnTransformer


def test_create_users_transformer(mongo):
    user_transformer = create_users_transformer()
    assert isinstance(user_transformer, ColumnTransformer)


def test_create_service_transformer(mongo):
    service_transformer = create_services_transformer()
    assert isinstance(service_transformer, ColumnTransformer)


def test_create_label_transformer(mongo):
    label_transformer = create_labels_transformer()
    assert isinstance(label_transformer, ColumnTransformer)


def test_create_transformer(mongo):
    user_transformer = create_transformer(USERS)
    service_transformer = create_transformer(SERVICES)
    label_transformer = create_transformer(LABELS)

    assert isinstance(user_transformer, ColumnTransformer)
    assert isinstance(service_transformer, ColumnTransformer)
    assert isinstance(label_transformer, ColumnTransformer)


def test_create_transformers(mongo):
    transformers = create_transformers()

    assert isinstance(transformers[USERS], ColumnTransformer)
    assert isinstance(transformers[SERVICES], ColumnTransformer)
    assert isinstance(transformers[LABELS], ColumnTransformer)


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
