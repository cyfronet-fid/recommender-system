# pylint: disable-all
import pickle

import torch
from pandas import DataFrame
from sklearn.compose import ColumnTransformer

from recommender.engine.pre_agent.preprocessing.dataframe_to_tensor import (
    df_to_tensor,
    raw_dataset_to_tensors,
    user_and_service_to_tensors,
    user_and_services_to_tensors,
    calculate_tensors_for_users_and_services,
)
from recommender.engine.pre_agent.preprocessing.mongo_to_dataframe import (
    service_to_df,
    create_raw_dataset,
    USERS,
    SERVICES,
    LABELS,
    calculate_dfs_for_users_and_services,
)
from recommender.engine.pre_agent.preprocessing.transformers import (
    create_services_transformer,
)
from recommender.models import User, Service
from tests.factories.marketplace import ServiceFactory, UserFactory


def test_df_to_tensor(mongo):
    service = ServiceFactory()
    df = service_to_df(service)
    services_transformer = create_services_transformer()
    tensor, services_transformer = df_to_tensor(df, services_transformer, fit=True)

    tensor2, _ = df_to_tensor(df, services_transformer, fit=False)

    assert isinstance(tensor, torch.Tensor)
    assert isinstance(tensor2, torch.Tensor)

    assert torch.all(torch.eq(tensor, tensor2))


def test_calculate_tensors_for_users_and_services(mongo):
    _no_one_services = [ServiceFactory() for _ in range(5)]
    common_services = [ServiceFactory() for _ in range(5)]

    user1 = UserFactory()
    user1.accessed_services = user1.accessed_services + common_services
    user1.save()

    user2 = UserFactory()
    user2.accessed_services = user2.accessed_services + common_services
    user2.save()

    raw_dataset = create_raw_dataset(save_df=True)
    raw_dataset_to_tensors(raw_dataset)

    for user in User.objects:
        assert user.tensor == []

    for service in Service.objects:
        assert service.tensor == []

    calculate_tensors_for_users_and_services()

    for user in User.objects:
        assert len(torch.Tensor(user.tensor).shape) == 1

    for service in Service.objects:
        assert len(torch.Tensor(service.tensor).shape) == 1


def test_raw_dataset_to_tensors(mongo):
    _no_one_services = [ServiceFactory() for _ in range(20)]
    common_services = [ServiceFactory() for _ in range(5)]

    user1 = UserFactory()
    user1.accessed_services = user1.accessed_services + common_services
    user1.save()

    user2 = UserFactory()
    user2.accessed_services = user2.accessed_services + common_services
    user2.save()

    raw_dataset = create_raw_dataset(save_df=True)
    tensors, transformers = raw_dataset_to_tensors(raw_dataset)

    assert isinstance(tensors[USERS], torch.Tensor)
    assert isinstance(tensors[SERVICES], torch.Tensor)
    assert isinstance(tensors[LABELS], torch.Tensor)

    assert tensors[USERS].shape[0] >= 20
    assert tensors[SERVICES].shape[0] >= 20
    assert tensors[LABELS].shape[0] >= 20

    assert isinstance(transformers[USERS], ColumnTransformer)
    assert isinstance(transformers[SERVICES], ColumnTransformer)
    assert isinstance(transformers[LABELS], ColumnTransformer)

    all_accessed_services = []
    for user in User.objects:
        all_accessed_services = all_accessed_services + user.accessed_services
        assert isinstance(pickle.loads(user.dataframe), DataFrame)

    for service in all_accessed_services:
        assert isinstance(pickle.loads(service.dataframe), DataFrame)


def test_user_and_services_to_tensors(mongo):
    _no_one_services = [ServiceFactory() for _ in range(20)]
    common_services = [ServiceFactory() for _ in range(5)]

    user1 = UserFactory()
    user1.accessed_services = user1.accessed_services + common_services
    user1.save()

    user2 = UserFactory()
    user2.accessed_services = user2.accessed_services + common_services
    user2.save()

    raw_dataset = create_raw_dataset()
    _, transformers = raw_dataset_to_tensors(raw_dataset)

    calculate_dfs_for_users_and_services()

    calculate_tensors_for_users_and_services(
        users_transformer=transformers[USERS],
        services_transformer=transformers[SERVICES],
    )

    u1 = User.objects[0]

    users_tensor, services_tensor = user_and_services_to_tensors(
        user=u1, services=u1.accessed_services
    )

    assert isinstance(users_tensor, torch.Tensor)
    assert users_tensor.shape[0] == len(u1.accessed_services)

    assert isinstance(services_tensor, torch.Tensor)
    assert services_tensor.shape[0] == len(u1.accessed_services)

    users_tensor, services_tensor = user_and_service_to_tensors(
        user=u1, service=u1.accessed_services[0]
    )

    assert isinstance(users_tensor, torch.Tensor)
    assert users_tensor.shape[0] == 1

    assert isinstance(services_tensor, torch.Tensor)
    assert services_tensor.shape[0] == 1
