# pylint: disable-all

import pytest
import torch
from torch.utils.data.dataset import Subset
from pandas import DataFrame
from sklearn.compose import ColumnTransformer

from recommender import User
from recommender.engines.autoencoders.training.data_extraction_step import (
    USERS,
    SERVICES,
    AUTOENCODERS,
)
from recommender.engines.autoencoders.training.data_preparation_step import (
    AEDataPreparationStep,
    DataPreparationStep,
    TRAIN,
    VALID,
    TEST,
    TRAIN_DS_SIZE,
    VALID_DS_SIZE,
    service_to_df,
    user_to_df,
    object_to_df,
    create_services_transformer,
    df_to_tensor,
    precalculate_tensors,
    create_transformer,
    precalc_users_and_service_tensors,
    create_users_transformer,
    validate_split,
)
from recommender.errors import InvalidObject
from recommender.models import Service
from tests.factories.marketplace import ServiceFactory, UserFactory
from tests.factories.populate_database import populate_users_and_services
from tests.conftest import users_services_args
from recommender.errors import InvalidDatasetSplit


def test_data_preparation_step(
    mongo, simulate_data_extraction_step, ae_pipeline_config
):
    """
    Testing:
    -> configuration
    -> split
    """
    ae_data_preparation_step = AEDataPreparationStep(ae_pipeline_config)
    # Check the correctness of the configuration inside data preparation step
    assert (
        ae_pipeline_config[DataPreparationStep.__name__]
        == ae_data_preparation_step.config
    )

    # Simulate the proper data extraction step
    data_ext_step, details_ext_step = simulate_data_extraction_step

    data_prep_step, details_prep_step = ae_data_preparation_step(data_ext_step)
    data_prep_step = data_prep_step[AUTOENCODERS]

    # Check the existence of a split
    for collection in data_prep_step.values():
        assert TRAIN and VALID and TEST in collection
        assert len(collection) == 3

    # Check class memberships
    for collection in (USERS, SERVICES):
        for split in (TRAIN, VALID, TEST):
            assert isinstance(data_prep_step[collection][split], Subset)

    # Check the correctness of a split
    # NOTE: Sometimes the split which we require cannot be exact in terms of integers (especially for small numbers)
    config = ae_pipeline_config[DataPreparationStep.__name__]
    split = {
        TRAIN: config[TRAIN_DS_SIZE],
        VALID: config[VALID_DS_SIZE],
        TEST: 1 - config[TRAIN_DS_SIZE] - config[VALID_DS_SIZE],
    }

    largest_collection = max(split, key=split.get)

    for collection in data_prep_step:
        assert len(data_prep_step[collection][largest_collection]) > 0


# Transformers
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


# Dataframes
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

    UserFactory.create_batch(3)
    ServiceFactory.create_batch(3)

    users = User.objects
    services = Service.objects

    for model_class in (users, services):
        for obj in model_class:
            assert obj.one_hot_tensor == []

    users_transformer = create_transformer(USERS)
    services_transformer = create_transformer(SERVICES)

    data = {
        USERS: [users, users_transformer],
        SERVICES: [services, services_transformer],
    }

    for model in data.values():
        tensors, fitted_users_transformer = precalculate_tensors(model[0], model[1])
        assert isinstance(tensors, torch.Tensor)
        assert isinstance(fitted_users_transformer, ColumnTransformer)

    for model_class in (users, services):
        for obj in model_class:
            assert len(obj.one_hot_tensor) > 0


def test_precalc_users_and_service_tensors(
    mongo, simulate_data_extraction_step, delete_users_services
):
    """
    Test precalc_users_and_service_tensors function
    -> with passed collection
    -> without passed collection
    """
    # With collection
    data, details = simulate_data_extraction_step
    data = data[AUTOENCODERS]

    for user in data[USERS]:
        assert user.one_hot_tensor == []
    for service in data[SERVICES]:
        assert service.one_hot_tensor == []

    tensors = precalc_users_and_service_tensors(collections=data)
    assert len(tensors[USERS]) > 0
    assert len(tensors[SERVICES]) > 0

    # Without collection
    args = users_services_args()
    args = list(args.values())
    populate_users_and_services(*args)

    for user in User.objects:
        assert user.one_hot_tensor == []

    for service in Service.objects:
        assert service.one_hot_tensor == []

    precalc_users_and_service_tensors(collections=None)

    for user in User.objects:
        assert len(user.one_hot_tensor) > 0

    for service in Service.objects:
        assert len(service.one_hot_tensor) > 0


def test_validate_split(simulate_data_preparation_step):
    """
    Test validate_split function
    -> with valid split
    -> with invalid split
    """
    data, _ = simulate_data_preparation_step
    data = data[AUTOENCODERS]
    for dataset in data.values():
        # Valid split
        validate_split(dataset)

    # Invalid split
    invalid_data = {
        USERS: {
            TRAIN: data[USERS][TRAIN],
            VALID: [],  # Valid should always have at lest one object
            TEST: None,
        },
        SERVICES: {TRAIN: data[USERS][TRAIN], VALID: [], TEST: None},
    }

    for dataset in invalid_data.values():
        with pytest.raises(InvalidDatasetSplit):
            validate_split(dataset)


def test_create_details(simulate_data_preparation_step):
    """
    Test create_details function which is called in simulate_data_preparation_step
    -> Correctness of the returned set
    """

    _, details = simulate_data_preparation_step

    # Check how many users and services were generated
    population = users_services_args(valid=True)
    services = population["common_services_num"] + population["unordered_services_num"]
    users = population["users_num"]
    objects_generated = {USERS: users, SERVICES: services}

    # Check how many users and services are in the details
    objects_sum = {USERS: 0, SERVICES: 0}

    for collection, datasets in details.items():
        assert TRAIN and VALID and TEST in datasets
        assert len(datasets) == 3

        for num_of_objects in datasets.values():
            assert isinstance(num_of_objects, int)
            objects_sum[collection] += num_of_objects

        assert objects_generated[collection] == objects_sum[collection]
