# pylint: disable=invalid-name, missing-class-docstring, missing-function-docstring, no-self-use
# pylint: disable=redefined-builtin, no-member, not-callable

"""Functions responsible for transforming Pandas dataframes
 to PyTorch tensors
"""
import pickle

import pandas as pd
import torch
from inflection import pluralize
from tqdm.auto import tqdm, trange

from recommender.engine.preprocessing.common import USERS, SERVICES
from recommender.engine.preprocessing.transformers import (
    create_transformer,
    save_transformer,
)
from recommender.models import User, Service

SERVICE_COLUMNS = [
    "name",
    "description",
    "tagline",
    "countries",
    "categories",
    "providers",
    "resource_organisation",
    "scientific_domains",
    "platforms",
    "target_users",
    "access_modes",
    "access_types",
    "trls",
    "life_cycle_statuses",
]

USER_COLUMNS = ["scientific_domains", "categories"]


def service_to_df(service, save_df=False):
    """It transform MongoEngine Service object into Pandas dataframe"""

    df = pd.DataFrame(columns=SERVICE_COLUMNS)

    row_dict = {
        "name": service.name,
        "description": service.description,
        "tagline": service.tagline.split(", "),
        "countries": list(service.countries),
        "categories": [category.name for category in service.categories],
        "providers": [provider.name for provider in service.providers],
        "resource_organisation": service.resource_organisation.name,
        "scientific_domains": [
            scientific_domain.name for scientific_domain in service.scientific_domains
        ],
        "platforms": [platform.name for platform in service.platforms],
        "target_users": [
            (target_user.name, target_user.description)
            for target_user in service.target_users
        ],
        "access_modes": [
            (access_mode.name, access_mode.description)
            for access_mode in service.access_modes
        ],
        "access_types": [
            (access_type.name, access_type.description)
            for access_type in service.access_types
        ],
        "trls": [(trl.name, trl.description) for trl in service.trls],
        "life_cycle_statuses": [
            (life_cycle_status.name, life_cycle_status.description)
            for life_cycle_status in service.life_cycle_statuses
        ],
    }

    df = df.append(row_dict, ignore_index=True)
    if save_df:
        service.dataframe = pickle.dumps(df)
        service.save()

    return df


def user_to_df(user, save_df=False):
    """It transform MongoEngine User object into Pandas dataframe"""

    df = pd.DataFrame(columns=USER_COLUMNS)

    row_dict = {
        "categories": [category.name for category in user.categories],
        "scientific_domains": [
            scientific_domain.name for scientific_domain in user.scientific_domains
        ],
    }

    df = df.append(row_dict, ignore_index=True)
    if save_df:
        user.dataframe = pickle.dumps(df)
        user.save()

    return df


def object_to_df(object, save_df=False):
    collection_name = pluralize(object.__class__.__name__.lower())

    if collection_name == USERS:
        object_df = user_to_df(object, save_df=save_df)
    elif collection_name == SERVICES:
        object_df = service_to_df(object, save_df=save_df)
    else:
        raise InvalidObject

    return object_df


def df_to_tensor(df, transformer, fit=False):
    """Transform Pandas dataframe into Pytorch one_hot_tensor using Scikit-learn
    transformer.
    """

    if fit:
        user_features_array = transformer.fit_transform(df)
    else:
        user_features_array = transformer.transform(df)

    tensor = torch.Tensor(user_features_array)

    return tensor, transformer


def precalculate_tensors(objects, transformer, fit=True, save=True):
    """Precalculate tensors for MongoEngine models"""

    objects = list(objects)
    collection_name = pluralize(objects[0].__class__.__name__.lower())
    if collection_name not in (USERS, SERVICES):
        raise InvalidObject

    # calculate and save dfs
    objects_df_rows = []
    name = pluralize(objects[0].__class__.__name__).lower()
    for object in tqdm(objects, desc=f"Calculating {name} dataframes..."):
        object_df = object_to_df(object, save_df=True)
        objects_df_rows.append(object_df)
    objects_df = pd.concat(objects_df_rows, ignore_index=True)

    # calculate and save tensors
    tensors, transformer = df_to_tensor(df=objects_df, transformer=transformer, fit=fit)

    for i in trange(len(objects), desc=f"Saving {name} tensors..."):
        objects[i].one_hot_tensor = tensors[i].tolist()
        objects[i].save()

    if fit and save:  # fit here for backwards compatibility
        save_transformer(transformer, collection_name)

    return transformer


def precalc_users_and_service_tensors():
    for model_class in (User, Service):
        name = pluralize(model_class.__name__).lower()
        t = create_transformer(name)
        t = precalculate_tensors(model_class.objects, t)
        save_transformer(t, name)


class NoPrecalculatedTensorsError(Exception):
    pass


def user_and_service_to_tensors(user, service):
    """It takes MongoEngine models from database and transform them into
    PyTorch tensors ready for inference.
    """

    if not (user.one_hot_tensor and service.one_hot_tensor):
        raise NoPrecalculatedTensorsError(
            "Given user or service has no precalculated one_hot_tensor"
        )

    users_ids = torch.tensor([user.id])

    user_tensor = torch.Tensor(user.one_hot_tensor)
    users_tensor = torch.unsqueeze(user_tensor, 0)

    services_ids = torch.tensor([service.id])

    service_tensor = torch.tensor(service.one_hot_tensor)
    services_tensor = torch.unsqueeze(service_tensor, 0)

    return users_ids, users_tensor, services_ids, services_tensor


def user_and_services_to_tensors(user, services):
    """Used for inferention in recommendation endpoint.
    It takes raw MongoEngine models of one user and related services
    and compose them into tensors ready for inference.
    """

    if not user.one_hot_tensor:
        raise NoPrecalculatedTensorsError(
            "Given user has no precalculated one_hot_tensor"
        )

    for service in services:
        if not service.one_hot_tensor:
            raise NoPrecalculatedTensorsError(
                "One or more of given services has/have no precalculated"
                " one_hot_tensor(s)"
            )

    services_ids = []
    services_tensors = []
    services = list(services)
    for service in services:
        services_ids.append(service.id)
        service_tensor = torch.unsqueeze(torch.Tensor(service.one_hot_tensor), dim=0)
        services_tensors.append(service_tensor)

    services_ids = torch.tensor(services_ids)
    services_tensor = torch.cat(services_tensors, dim=0)

    n = services_tensor.shape[0]
    users_ids = torch.full([n], user.id)
    user_tensor = torch.Tensor(user.one_hot_tensor)
    users_tensor = torch.unsqueeze(user_tensor, dim=0).repeat(n, 1)

    return users_ids, users_tensor, services_ids, services_tensor


class InvalidObject(Exception):
    def message(self):  # pragma: no cover
        return "Invalid object (should be 'User' or 'Service' instance)"
