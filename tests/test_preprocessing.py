# pylint: disable-all
import pytest
import torch
from pandas import DataFrame
from sklearn.compose import ColumnTransformer

from recommender.engine.pre_agent.preprocessing import USERS, SERVICES
from recommender.engine.pre_agent.preprocessing.preprocessing import (
    service_to_df,
    user_to_df,
    object_to_df,
    df_to_tensor,
    precalc_users_and_service_tensors,
    user_and_services_to_tensors,
    user_and_service_to_tensors,
    InvalidObject,
    precalculate_tensors,
    NoPrecalculatedTensorsError,
)
from recommender.engine.pre_agent.preprocessing.transformers import (
    create_services_transformer,
    create_transformer,
    create_users_transformer,
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
        assert user.tensor == []

    users_transformer = create_transformer(USERS)
    fitted_users_transformer = precalculate_tensors(users, users_transformer)

    assert isinstance(fitted_users_transformer, ColumnTransformer)

    for user in User.objects:
        assert len(user.tensor) > 0

    # Services
    ServiceFactory.create_batch(3)
    services = Service.objects

    for service in Service.objects:
        assert service.tensor == []

    services_transformer = create_transformer(SERVICES)
    fitted_services_transformer = precalculate_tensors(services, services_transformer)

    assert isinstance(fitted_services_transformer, ColumnTransformer)

    for service in Service.objects:
        assert len(service.tensor) > 0


def test_precalc_users_and_service_tensors(mongo):
    populate_users_and_services(
        common_services_number=4,
        no_one_services_number=1,
        users_number=4,
        k_common_services_min=1,
        k_common_services_max=3,
    )

    for user in User.objects:
        assert user.tensor == []

    for service in Service.objects:
        assert service.tensor == []

    precalc_users_and_service_tensors()

    for user in User.objects:
        assert len(user.tensor) > 0

    for service in Service.objects:
        assert len(service.tensor) > 0


def test_user_and_services_to_tensors(mongo):
    populate_users_and_services(
        common_services_number=4,
        no_one_services_number=1,
        users_number=4,
        k_common_services_min=1,
        k_common_services_max=3,
    )

    with pytest.raises(NoPrecalculatedTensorsError):
        u1 = User.objects[0]
        user_and_service_to_tensors(user=u1, service=u1.accessed_services[0])

    with pytest.raises(NoPrecalculatedTensorsError):
        u1 = User.objects[0]
        user_and_services_to_tensors(user=u1, services=u1.accessed_services)

    with pytest.raises(NoPrecalculatedTensorsError):
        u1 = User.objects[0]
        precalculate_tensors([u1], create_users_transformer())
        user_and_services_to_tensors(user=u1, services=u1.accessed_services)

    precalc_users_and_service_tensors()

    u1 = User.objects[0]

    (
        users_ids,
        users_tensor,
        services_ids,
        services_tensor,
    ) = user_and_service_to_tensors(user=u1, service=u1.accessed_services[0])

    assert isinstance(users_ids, torch.Tensor)
    assert len(users_ids.shape) == 1
    assert users_ids.shape[0] == 1

    assert isinstance(users_tensor, torch.Tensor)
    assert len(users_tensor.shape) == 2
    assert users_tensor.shape[0] == 1

    assert isinstance(services_ids, torch.Tensor)
    assert len(services_ids.shape) == 1
    assert services_ids.shape[0] == 1

    assert isinstance(services_tensor, torch.Tensor)
    assert len(services_tensor.shape) == 2
    assert services_tensor.shape[0] == 1

    (
        users_ids,
        users_tensor,
        services_ids,
        services_tensor,
    ) = user_and_services_to_tensors(user=u1, services=u1.accessed_services)

    assert isinstance(users_ids, torch.Tensor)
    assert len(users_ids.shape) == 1
    assert users_ids.shape[0] == len(u1.accessed_services)

    assert isinstance(users_tensor, torch.Tensor)
    assert len(users_tensor.shape) == 2
    assert users_tensor.shape[0] == len(u1.accessed_services)

    assert isinstance(services_ids, torch.Tensor)
    assert len(services_ids.shape) == 1
    assert services_ids.shape[0] == len(u1.accessed_services)

    assert isinstance(services_tensor, torch.Tensor)
    assert len(services_tensor.shape) == 2
    assert services_tensor.shape[0] == len(u1.accessed_services)
