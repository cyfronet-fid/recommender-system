# pylint: disable=invalid-name, missing-class-docstring, missing-function-docstring, no-self-use
# pylint: disable=redefined-builtin, no-member

"""Functions responsible for transforming Pandas dataframes
 to PyTorch tensors
"""
import pickle

import torch
import pandas as pd
from pandas import DataFrame
from tqdm.auto import tqdm

from recommender.engine.pre_agent.preprocessing.mongo_to_dataframe import (
    USERS,
    SERVICES,
    LABELS,
    user_to_df,
    service_to_df,
)
from recommender.engine.pre_agent.preprocessing.transformers import (
    create_users_transformer,
    create_transformers,
    create_transformer,
    save_transformer,
    load_last_transformer,
)
from recommender.models import User, Service


def df_to_tensor(df, transformer, fit=False):
    """Transform Pandas dataframe into Pytorch tensor using Scikit-learn
    transformer.
    """

    if transformer is None:
        transformer = create_users_transformer()
        fit = True

    if fit:
        user_features_array = transformer.fit_transform(df)
    else:
        user_features_array = transformer.transform(df)

    tensor = torch.Tensor(user_features_array)

    return tensor, transformer


class NoPrecalculatedDataframeError(Exception):
    def message(self):
        return "object should have precalculated dataframe"


class InvalidPrecalculatedDataframeError(Exception):
    def message(self):
        return "Invalid precalculated dataframe"


def _calculate_tensors_for_objects(objects, objects_transformer):
    """Calculate tensors for given objects using given transformers and
    based on calculated dataframes.
    """
    objects_dfs = []
    for object in tqdm(objects):
        if object.dataframe is None:
            raise NoPrecalculatedDataframeError

        object_df = pickle.loads(object.dataframe)
        if not isinstance(object_df, DataFrame):
            raise InvalidPrecalculatedDataframeError

        objects_dfs.append(object_df)

    objects_df = pd.concat(objects_dfs, ignore_index=True)
    objects_tensors, _ = df_to_tensor(objects_df, objects_transformer)
    for i, object_tensor in tqdm(enumerate(objects_tensors)):
        objects[i].tensor = object_tensor.tolist()
        objects[i].save()


def calculate_tensors_for_users_and_services(
    users_transformer=None, services_transformer=None
):
    """Calculate tensors for all users and services using given transformers
    and based on calculated dataframes.
    """
    if users_transformer is None:
        users_transformer = load_last_transformer(USERS)

    if services_transformer is None:
        services_transformer = load_last_transformer(SERVICES)

    user_objects = list(User.objects)
    service_objects = list(Service.objects)

    _calculate_tensors_for_objects(user_objects, users_transformer)
    _calculate_tensors_for_objects(service_objects, services_transformer)


def raw_dataset_to_tensors(raw_dataset, transformers=None, fit=False):
    """Transform raw Pandas dataframe dataset into dict of Pytorch tensors

    transformers
    dict of transformers
    If transformers are not provided then they are created,
     fitted and used for transforming.
    If transformers are provided then they are not fitted by default but only
     used for transforming.
    If transformers are partially provided missing transformers are created
     and fitted while provided transformers are not fitted by default.

    fit
    can be bool or dict
    If bool then all provided transformers are fitted or not depending on fit
     value. Not provided transformers are always fitted.
    If dict then all specified transformers are fitted or not depending on
     its value in the fit dict. Not provided transformers are always fitted.

    It returns tensors and fitted transformers. Moreover transformers are
     saved to database using standard names.
    """

    if isinstance(fit, bool):
        fit = {USERS: fit, SERVICES: fit, LABELS: fit}

    keys = [USERS, SERVICES, LABELS]

    if transformers is None:
        transformers = create_transformers()
        for key in keys:
            fit[key] = True

    for key in keys:
        if transformers.get(key) is None:
            transformers[key] = create_transformer(key)
            fit[key] = True

    tensors = {}
    updated_transformers = {}
    for key in keys:
        tensor, transformer = df_to_tensor(
            df=raw_dataset[key], transformer=transformers[key], fit=fit[key]
        )
        tensors[key] = tensor
        updated_transformers[key] = transformer
        save_transformer(transformer, key)

    return tensors, updated_transformers


def user_and_service_to_tensors(
    user, service, users_transformer=None, services_transformer=None
):
    """Used for inferention. It takes raw MongoEngine
    models from database, transform them into Pandas dataframes and uses
    transformers fitted before on the entire dataset to transform
    dataframes into PyTorch tensors.

    If transformers are not provided then last transformers are loaded from
     database.
    """

    if user.tensor and service.tensor:
        user_tensor = torch.Tensor(user.tensor)
        users_tensor = torch.unsqueeze(user_tensor, 0)

        service_tensor = torch.Tensor(service.tensor)
        services_tensor = torch.unsqueeze(service_tensor, 0)

        return users_tensor, services_tensor

    if users_transformer is None:
        users_transformer = load_last_transformer(USERS)

    if services_transformer is None:
        services_transformer = load_last_transformer(SERVICES)

    user_df = user_to_df(user)
    service_df = service_to_df(service)

    users_tensor, _ = df_to_tensor(user_df, users_transformer)
    services_tensor, _ = df_to_tensor(service_df, services_transformer)

    return users_tensor, services_tensor


def _services_to_df(services):
    services_dfs = []
    for service in services:
        service_df = service_to_df(service)
        services_dfs.append(service_df)
    services_df = pd.concat(services_dfs, ignore_index=True)

    return services_df


def _user_to_repeated_df(user, amount):
    user_df = user_to_df(user)
    users_df = pd.DataFrame(
        user_df.values.repeat(amount, axis=0), columns=user_df.columns
    )

    return users_df


def user_and_services_to_tensors(user, services):
    """Used for inferention in recommendation endpoint.
    It takes raw MongoEngine models of one user and related services
    and compose them into tensors ready for inference using precalculated tensors.
    """

    services_tensors = []
    services = list(services)
    for service in services:
        service_tensor = torch.unsqueeze(torch.Tensor(service.tensor), dim=0)
        services_tensors.append(service_tensor)

    services_tensor = torch.cat(services_tensors, dim=0)

    user_tensor = torch.Tensor(user.tensor)
    n = services_tensor.shape[0]
    users_tensor = torch.unsqueeze(user_tensor, dim=0).repeat(n, 1)

    return users_tensor, services_tensor
