# pylint: disable-all
import pytest
import torch
from pandas import DataFrame
from sklearn.compose import ColumnTransformer

from recommender.engines.autoencoders.training.data_extraction_step import (
    USERS,
    SERVICES,
)
from recommender.errors import InvalidObject

from recommender.engines.autoencoders.training.data_preparation_step import (
    create_services_transformer,
    create_transformer,
    service_to_df,
    user_to_df,
    object_to_df,
    df_to_tensor,
    precalculate_tensors,
    precalc_users_and_service_tensors,
)
from recommender.models import User, Service
from tests.factories.populate_database import populate_users_and_services
from tests.factories.marketplace import ServiceFactory, UserFactory


def test_service_to_df(mongo):
    service = ServiceFactory()
    df = service_to_df(service)
    assert isinstance(df, DataFrame)


def test_user_to_df(mongo):
    user = UserFactory()
    df = user_to_df(user)
    assert isinstance(df, DataFrame)


def test_object_to_df(mongo):
    user = UserFactory()
    user_df = object_to_df(user)
    assert isinstance(user_df, DataFrame)

    service = UserFactory()
    service_df = object_to_df(service)
    assert isinstance(service_df, DataFrame)

    with pytest.raises(InvalidObject):
        object_to_df("placeholder_object")


def test_df_to_tensor(mongo):
    service = ServiceFactory()
    df = service_to_df(service)
    services_transformer = create_services_transformer()
    tensor, services_transformer = df_to_tensor(df, services_transformer, fit=True)

    tensor2, _ = df_to_tensor(df, services_transformer, fit=False)

    assert isinstance(tensor, torch.Tensor)
    assert isinstance(tensor2, torch.Tensor)

    assert torch.all(torch.eq(tensor, tensor2))


def test_precalculate_tensors(mongo):
    # Invalid objects
    placeholder_objects = ["placeholder_object"]
    with pytest.raises(InvalidObject):
        precalculate_tensors(placeholder_objects, None)

    # Users
    users = UserFactory.create_batch(3)

    for user in User.objects:
        assert user.one_hot_tensor == []

    users_transformer = create_transformer(USERS)
    tensors, fitted_users_transformer = precalculate_tensors(users, users_transformer)

    assert isinstance(fitted_users_transformer, ColumnTransformer)

    for user in User.objects:
        assert len(user.one_hot_tensor) > 0

    # Services
    ServiceFactory.create_batch(3)
    services = Service.objects

    for service in Service.objects:
        assert service.one_hot_tensor == []

    services_transformer = create_transformer(SERVICES)
    tensor, fitted_services_transformer = precalculate_tensors(
        services, services_transformer
    )

    assert isinstance(fitted_services_transformer, ColumnTransformer)

    for service in Service.objects:
        assert len(service.one_hot_tensor) > 0


def test_precalc_users_and_service_tensors(mongo):
    populate_users_and_services(
        common_services_no=4,
        unordered_services_no=1,
        total_users=4,
        k_common_services_min=1,
        k_common_services_max=3,
    )

    for user in User.objects:
        assert user.one_hot_tensor == []

    for service in Service.objects:
        assert service.one_hot_tensor == []

    precalc_users_and_service_tensors()

    for user in User.objects:
        assert len(user.one_hot_tensor) > 0

    for service in Service.objects:
        assert len(service.one_hot_tensor) > 0
