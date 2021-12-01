# pylint: disable=invalid-name, missing-class-docstring, missing-function-docstring, no-self-use
# pylint: disable=redefined-builtin, no-member, not-callable
# pylint: disable=line-too-long, no-else-return, fixme

"""Autoencoder Data Preparation Step."""

import pickle
from typing import Tuple

import pandas as pd
import torch
from inflection import pluralize
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from torch.utils.data import random_split
from tqdm.auto import tqdm, trange

from recommender.errors import (
    InvalidObject,
)

from recommender.engines.autoencoders.ml_components.list_column_one_hot_encoder import (
    ListColumnOneHotEncoder,
)
from recommender.engines.base.base_steps import DataPreparationStep
from recommender.engines.autoencoders.training.data_extraction_step import (
    AUTOENCODERS,
    USERS,
    SERVICES,
)
from recommender.engines.constants import DEVICE
from recommender.models import User, Service
from recommender.errors import InvalidDatasetSplit

TRAIN = "training"
VALID = "validation"
TEST = "testing"
TRAIN_DS_SIZE = "train_ds_size"
VALID_DS_SIZE = "valid_ds_size"
DATASETS = "datasets"

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


def create_users_transformer():
    """Creates users transformer"""

    transformer = make_column_transformer(
        (make_pipeline(ListColumnOneHotEncoder()), ["scientific_domains", "categories"])
    )

    return transformer


def create_services_transformer():
    """Creates services transformer"""

    transformer = make_column_transformer(
        (
            make_pipeline(ListColumnOneHotEncoder()),
            [
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
            ],
        )
    )

    return transformer


def create_transformer(name):
    """Creates new transformer of given name."""

    if name == USERS:
        return create_users_transformer()
    elif name == SERVICES:
        return create_services_transformer()

    raise ValueError


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
    """Transform object into Pandas dataframe"""
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


def precalculate_tensors(objects, transformer, fit=True, verbose=False):
    """Precalculate tensors for MongoEngine models"""

    objects = list(objects)
    collection_name = pluralize(objects[0].__class__.__name__.lower())
    if collection_name not in (USERS, SERVICES):
        raise InvalidObject

    # calculate and save dfs
    objects_df_rows = []
    name = pluralize(objects[0].__class__.__name__).lower()

    for obj in tqdm(
        objects, desc=f"Calculating {name} dataframes...", disable=(not verbose)
    ):
        object_df = object_to_df(obj, save_df=True)
        objects_df_rows.append(object_df)
    objects_df = pd.concat(objects_df_rows, ignore_index=True)

    # calculate and save tensors
    tensors, transformer = df_to_tensor(df=objects_df, transformer=transformer, fit=fit)

    for i in trange(
        len(objects), desc=f"Saving {name} tensors...", disable=(not verbose)
    ):
        objects[i].one_hot_tensor = tensors[i].tolist()
        objects[i].save()

    return tensors, transformer


def precalc_users_and_service_tensors(collections: dict = None):
    """
    Precalculate users and services tensors

    Args:
        collections: a dictionary of User and Service objects from data extraction step.
                     If not specified, the function takes User and Service objects from a database.
    """
    tensors = {USERS: {}, SERVICES: {}}

    if collections:
        data = collections.items()
    else:
        data = {}
        for model_class in (User, Service):
            name = pluralize(model_class.__name__).lower()
            objects = model_class.objects
            data[name] = objects
        data = data.items()

    for name, obj in data:
        t = create_transformer(name)
        tensor, _ = precalculate_tensors(obj, t)
        if name == USERS:
            tensors[USERS] = tensor
        elif name == SERVICES:
            tensors[SERVICES] = tensor
        else:
            raise ValueError

    return tensors


def split_autoencoder_datasets(
    collection,
    train_ds_size=0.6,
    valid_ds_size=0.2,
    device: torch.device = torch.device("cpu"),
):
    """Split users/services autoencoder dataset into train/valid/test datasets"""

    ds_tensor = collection.to(device)
    dataset = torch.utils.data.TensorDataset(ds_tensor)

    ds_size = len(dataset)
    train_ds_size = int(train_ds_size * ds_size)
    valid_ds_size = int(valid_ds_size * ds_size)
    test_ds_size = int(ds_size - (train_ds_size + valid_ds_size))

    train_ds, valid_ds, test_ds = random_split(
        dataset, [train_ds_size, valid_ds_size, test_ds_size]
    )

    output = {TRAIN: train_ds, VALID: valid_ds, TEST: test_ds}

    return output


def validate_split(datasets):
    """Train and valid datasets should always have at least one object"""
    for split in (TRAIN, VALID):
        if len(datasets[split]) < 1:
            print(
                f"{split} dataset of user or service is empty after the split"
            )  # TODO logger
            raise InvalidDatasetSplit()


def create_details(data):
    """Count objects in each train/valid/test dataset for User/Service collections"""
    details = {USERS: {}, SERVICES: {}}
    data = data[AUTOENCODERS]

    for collection, datasets in data.items():
        for split in datasets:
            details[collection][split] = len(split)

    return details


class AEDataPreparationStep(DataPreparationStep):
    """Autoencoder data preparation step"""

    def __init__(self, pipeline_config):
        super().__init__(pipeline_config)
        self.device = self.resolve_constant(DEVICE, torch.device("cpu"))
        self.train_ds_size = self.resolve_constant(TRAIN_DS_SIZE, 0.6)
        self.valid_ds_size = self.resolve_constant(VALID_DS_SIZE, 0.2)

    def __call__(self, data: dict = None) -> Tuple[object, dict]:
        all_datasets = {AUTOENCODERS: {}}
        raw_data = data[AUTOENCODERS]

        tensors = precalc_users_and_service_tensors(raw_data)

        for collection_name, dataset in tensors.items():
            splitted_ds = split_autoencoder_datasets(
                dataset,
                train_ds_size=self.train_ds_size,
                valid_ds_size=self.valid_ds_size,
                device=self.device,
            )

            validate_split(splitted_ds)
            all_datasets[AUTOENCODERS][collection_name] = splitted_ds

        data = all_datasets
        details = create_details(data)

        return data, details
