# pylint: disable=no-member, not-callable

"""This module contains function that creates datasets needed for Neural
 Collaborative filtering model training"""

import random
from copy import deepcopy

import numpy as np
import torch
from tqdm.auto import tqdm

from recommender.engine.pre_agent.models import NEURAL_CF
from recommender.engine.pre_agent.datasets.common import TRAIN, VALID, TEST
from recommender.engine.pre_agent.datasets.tensor_dict_dataset import TensorDictDataset
from recommender.engine.pre_agent.preprocessing import USERS, SERVICES, LABELS
from recommender.models import Service, User

USERS_IDS = "users_ids"
SERVICES_IDS = "services_ids"


def get_ncf_dataset_name(split):
    """Get the name of the Neural Collaborative filtering dataset"""

    valid_splits = (TRAIN, VALID, TEST)
    if split not in valid_splits:
        raise ValueError(f"Invalid split, should be one of: {valid_splits}")

    return f"{NEURAL_CF} {split} dataset"


def split_services(ordered_services, train_ds_size=0.6, valid_ds_size=0.2):
    """Split ordered services using provided percentages and taking into
    consideration time:
    services treated as train set are ordered before services treated
     as validation set and services treated as test set are ordered
     after services treated as validation set"""

    ordered_services = list(ordered_services)
    ds_size = len(ordered_services)
    test_ds_size = 1 - (train_ds_size + valid_ds_size)
    train_ds_size = int(np.ceil(train_ds_size * ds_size))
    test_ds_size = int(np.floor(test_ds_size * ds_size))
    valid_ds_size = ds_size - (train_ds_size + test_ds_size)

    train_ordered_services = ordered_services[:train_ds_size]
    valid_ordered_services = ordered_services[
        train_ds_size : train_ds_size + valid_ds_size
    ]
    test_ordered_services = ordered_services[train_ds_size + valid_ds_size :]

    output = {
        TRAIN: train_ordered_services,
        VALID: valid_ordered_services,
        TEST: test_ordered_services,
    }

    return output


def get_not_ordered_services(ordered_services):
    """Given ordered services find not ordered services"""

    ordered_services = list(ordered_services)
    ordered_services_ids = [s.id for s in ordered_services]
    all_not_ordered_services = list(Service.objects(id__nin=ordered_services_ids))
    k = min(len(ordered_services), len(all_not_ordered_services))
    not_ordered_services = random.sample(all_not_ordered_services, k=k)

    return not_ordered_services


def set_data(datasets, user, services, split_type, class_tensor):
    """Update datasets dict for given arguments"""

    for service in services[split_type]:
        datasets[split_type][USERS].append(
            torch.unsqueeze(torch.Tensor(user.tensor), dim=0)
        )
        datasets[split_type][USERS_IDS].append(user.id)
        datasets[split_type][SERVICES].append(
            torch.unsqueeze(torch.Tensor(service.tensor), dim=0)
        )
        datasets[split_type][SERVICES_IDS].append(service.id)
        datasets[split_type][LABELS].append(torch.unsqueeze(class_tensor, dim=0))


def create_ncf_datasets(
    train_ds_size=0.6, valid_ds_size=0.2, max_users=None, device=torch.device("cpu")
):
    """Creates balanced dataset that consist of pairs user-service.

    If there is n users and each of them has on average k services
    then the final dataset will consist of 2kn examples
    (not just kn because for each k positive examples of services
    oredered by a user there are generated also k negative services
    not ordered by a user).

    Time and space complexity of this algorithm is O(kn)
    """

    dataset = {USERS: [], USERS_IDS: [], SERVICES: [], SERVICES_IDS: [], LABELS: []}

    split_types = [TRAIN, VALID, TEST]

    datasets = {key: deepcopy(dataset) for key in split_types}

    ordered_class_tensor = torch.Tensor([1.0])
    not_ordered_class_tensor = torch.Tensor([0.0])

    users = list(User.objects)
    if max_users is not None:
        users = users[:max_users]

    for user in tqdm(users, desc="Generating dataset..."):
        ordered_services = user.accessed_services
        not_ordered_services = deepcopy(
            get_not_ordered_services(ordered_services)
        )  # (same amount as positive - classes balance)

        ordered_services = split_services(
            ordered_services, train_ds_size, valid_ds_size
        )

        not_ordered_services = split_services(
            not_ordered_services, train_ds_size, valid_ds_size
        )

        for split_type in split_types:
            # Positive examples
            set_data(datasets, user, ordered_services, split_type, ordered_class_tensor)

            # Negative examples
            set_data(
                datasets,
                user,
                not_ordered_services,
                split_type,
                not_ordered_class_tensor,
            )

    for split_type in split_types:
        if datasets[split_type][USERS]:
            for kind in [USERS, SERVICES, LABELS]:
                datasets[split_type][kind] = torch.cat(
                    datasets[split_type][kind], dim=0
                ).to(device)
        else:
            for kind in [USERS, SERVICES, LABELS]:
                datasets[split_type][kind] = torch.tensor([]).to(device)

        datasets[split_type][USERS_IDS] = torch.tensor(
            datasets[split_type][USERS_IDS]
        ).to(device)
        datasets[split_type][SERVICES_IDS] = torch.tensor(
            datasets[split_type][SERVICES_IDS]
        ).to(device)

        datasets[split_type] = TensorDictDataset(datasets[split_type])

    return datasets
