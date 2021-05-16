# pylint: disable=no-member

"""This module contain implementation of creation all needed datasets for
 pre-agent training"""

import torch

from recommender.engine.utils import save_dataset
from recommender.engine.agents.pre_agent.models import NEURAL_CF
from recommender.engine.datasets.autoencoders import (
    AUTOENCODERS,
    create_autoencoder_datasets,
)
from recommender.engine.agents.pre_agent.datasets.neural_collaborative_filtering import (
    create_ncf_datasets,
)
from recommender.engine.preprocessing import USERS, SERVICES
from settings import get_device


def create_datasets():
    """Creates Pre Agent Dataset, split it into train/valid/test datasets and
    saves each of them."""

    device_name = get_device("TRAINING_DEVICE")
    device = torch.device(device_name)

    all_datasets = {AUTOENCODERS: {}}

    for collection_name in (USERS, SERVICES):
        print(f"Creating {collection_name} autoencoder datasets...")
        datasets = create_autoencoder_datasets(collection_name, device=device)
        print(f"{collection_name} autoencoder datasets created successfully!")

        all_datasets[AUTOENCODERS][collection_name] = datasets

        print(f"Saving {collection_name} autoencoder datasets...")
        for split, dataset in datasets.items():
            save_dataset(
                dataset, name=f"{AUTOENCODERS} {collection_name} {split} dataset"
            )
        print(f"{collection_name} autoencoder datasets saved successfully!")

    print(f"Creating {NEURAL_CF} datasets...")
    datasets = create_ncf_datasets(device=device)
    print(f"{NEURAL_CF} datasets created successfully!")

    all_datasets[NEURAL_CF] = datasets

    print(f"Saving {NEURAL_CF} datasets...")
    for split, dataset in datasets.items():
        save_dataset(dataset, name=f"{NEURAL_CF} {split} dataset")
    print(f"{NEURAL_CF} datasets saved successfully!")

    return all_datasets
