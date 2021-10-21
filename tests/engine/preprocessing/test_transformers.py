# pylint: disable-all

import pytest
from sklearn.compose import ColumnTransformer

from recommender.engines.autoencoders.training.data_extraction_step import (
    USERS,
    SERVICES,
)
from recommender.engines.autoencoders.training.data_preparation_step import (
    create_users_transformer,
    create_services_transformer,
    create_transformer,
)


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
