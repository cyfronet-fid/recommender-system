# pylint: disable=too-many-arguments, fixme

"""Neural Collaborative Filtering Data Preparation Step."""

from copy import deepcopy
from typing import Tuple, Dict, List, Union

import numpy as np
import torch
from mongoengine import QuerySet
from torch import Tensor

from recommender.engine.preprocessing.embedder import (
    Embedder,
)  # TODO: import embedder from proper module
from recommender.engine.preprocessing.normalizer import Normalizer
from recommender.engines.base.base_steps import DataPreparationStep
from recommender.engines.constants import DEVICE
from recommender.engines.ncf.training.data_extraction_step import (
    USER,
    ORDERED_SERVICES,
    NOT_ORDERED_SERVICES,
    RAW_DATA,
    USERS,
    SERVICES,
)
from recommender.engines.ncf.ml_components.tensor_dict_dataset import (
    TensorDictDataset,
)
from recommender.models import User, Service

TRAIN_DS_SIZE = "train_ds_size"
VALID_DS_SIZE = "valid_ds_size"

ORDERED_LABEL = 1.0
NOT_ORDERED_LABEL = 0.0

TRAIN = "train"
VALID = "valid"
TEST = "test"

LABELS = "labels"
USERS_IDS = "users_ids"
SERVICES_IDS = "services_ids"
DATASETS = "datasets"
EXAMPLE_NUMBERS = "example_numbers"


def embed(
    data: Dict[str, Union[List, int, QuerySet]]
) -> Dict[str, Union[list, int, QuerySet]]:
    """Embed users and services in provided data using Embedders.

    Args:
        data: data produced in NCFDataExtractionStep.__call__.

    Returns:
        data: reloaded data (with new tensors inside objects).

    """

    user_embedder = Embedder.load(version="user")  # TODO: use constant from Embedders
    user_embedder(data[USERS], use_cache=False, save_cache=True)
    services_embedder = Embedder.load(
        version="service"
    )  # TODO: use constant from Embedders
    services_embedder(data[SERVICES], use_cache=False, save_cache=True)

    for entry in data[RAW_DATA]:
        entry[USER].reload()
        for kind in (ORDERED_SERVICES, NOT_ORDERED_SERVICES):
            for service in entry[kind]:
                service.reload()

    return data


def normalise(data: Dict[str, Union[list, int, QuerySet]]) -> List[Dict[str, Dict]]:
    """Normalise users and services in provided data using universal Normalizer.

    Args:
        data: data produced in NCFDataExtractionStep.__call__.

    Returns:
        data: reloaded raw_data (with new tensors inside objects).
    """
    raw_data = data[RAW_DATA]

    normalizer = Normalizer()
    normalizer(data[USERS], save_cache=True)
    normalizer(data[SERVICES], save_cache=True)

    for entry in raw_data:
        entry[USER].reload()
        for kind in (ORDERED_SERVICES, NOT_ORDERED_SERVICES):
            for service in entry[kind]:
                service.reload()

    return raw_data


def _split_services(
    ordered_services: List[Service], train_ds_size: float, valid_ds_size: float
) -> Dict[str, List[Service]]:
    """Split ordered services using provided percentages and taking into
    consideration time:
    services treated as train set are ordered before services treated
     as validation set and services treated as test set are ordered
     after services treated as validation set."""

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


def split_and_group(
    datasets: Dict[str, Dict[str, List]],
    user: User,
    services: List[Service],
    label: float,
    train_ds_size: float,
    valid_ds_size: float,
):
    """Split and group services into proper dict form."""
    services_splits = _split_services(services, train_ds_size, valid_ds_size)
    for split_name, services_split in services_splits.items():
        dataset = datasets[split_name]
        for service in services_split:
            dataset[USERS].append(user.dense_tensor)
            dataset[USERS_IDS].append(user.id)
            dataset[SERVICES].append(service.dense_tensor)
            dataset[SERVICES_IDS].append(service.id)
            dataset[LABELS].append(label)


def prepare_empty_datasets() -> Dict:
    """Prepare empty dataset in the form of nested dicts."""

    splits = [TRAIN, TEST, VALID]
    dataset = {USERS: [], SERVICES: [], USERS_IDS: [], SERVICES_IDS: [], LABELS: []}

    datasets = {split: deepcopy(dataset) for split in splits}

    return datasets


def tensorize(
    data: List[Dict], train_ds_size: float, valid_ds_size: float
) -> Dict[str, Dict[str, Tensor]]:
    """Tensorize data using proper split percentages."""

    datasets = prepare_empty_datasets()
    for entry in data:
        user = entry[USER]
        ordered_services = entry[ORDERED_SERVICES]
        not_ordered_services = entry[NOT_ORDERED_SERVICES]
        split_and_group(
            datasets,
            user,
            ordered_services,
            ORDERED_LABEL,
            train_ds_size,
            valid_ds_size,
        )
        split_and_group(
            datasets,
            user,
            not_ordered_services,
            NOT_ORDERED_LABEL,
            train_ds_size,
            valid_ds_size,
        )

    tensors_dict = prepare_empty_datasets()

    for split_name, dataset in datasets.items():
        for kind in (USERS, SERVICES):
            tensors_dict[split_name][kind] = torch.Tensor(dataset[kind])
        for kind in (USERS_IDS, SERVICES_IDS):
            tensors_dict[split_name][kind] = torch.tensor(dataset[kind])
        tensors_dict[split_name][LABELS] = torch.Tensor(dataset[LABELS]).reshape(
            (-1, 1)
        )
    return tensors_dict


def cast_to_device(
    tensors_dict: Dict[str, Dict[str, Tensor]], device: torch.device
) -> Dict[str, Dict[str, Tensor]]:
    """cast datasets to the device."""

    for dataset in tensors_dict.values():
        for name, tensor in dataset.items():
            dataset[name] = tensor.to(device)
    return tensors_dict


def create_pytorch_datasets(
    tensors_dict: Dict[str, Dict[str, Tensor]]
) -> Dict[str, TensorDictDataset]:
    """Create pytorch tensor-dict datasets out of dicts of tensors."""
    new_tensor_dict = {}
    for split_name, dataset in tensors_dict.items():
        new_tensor_dict[split_name] = TensorDictDataset(dataset)

    return new_tensor_dict


class NCFDataPreparationStep(DataPreparationStep):
    """Neural Collaborative Filtering Data Preparation Step."""

    def __init__(self, pipeline_config):
        super().__init__(pipeline_config)
        self.train_ds_size = self.resolve_constant(TRAIN_DS_SIZE, 0.6)
        self.valid_ds_size = self.resolve_constant(VALID_DS_SIZE, 0.2)
        self.device = self.resolve_constant(DEVICE, torch.device("cpu"))

    def __call__(self, data: Dict = None) -> Tuple[Dict, Dict]:
        """Perform data preparation consisting of:
        -> embedding,
        -> normalisation,
        -> casting to the proper device,
        -> tensorization into pytorch dataset.
        """

        embedded_data = embed(data)
        normalised_data = normalise(embedded_data)
        tensors_dict = tensorize(
            normalised_data, self.train_ds_size, self.valid_ds_size
        )
        casted_tensors_dict = cast_to_device(tensors_dict, self.device)
        pytorch_datasets = create_pytorch_datasets(casted_tensors_dict)

        data.pop(RAW_DATA)
        data[DATASETS] = pytorch_datasets

        details = {
            EXAMPLE_NUMBERS: {
                ds_name: len(pytorch_datasets[ds_name])
                for ds_name in (TRAIN, TEST, VALID)
            }
        }

        return data, details
