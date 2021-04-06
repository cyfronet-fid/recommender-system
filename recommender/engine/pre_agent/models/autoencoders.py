# pylint: disable=no-member, too-many-arguments, missing-function-docstring

"""This module contains Autoencoders models instantiating-related functions"""

from copy import deepcopy

import torch
from torch.nn import Module, Sequential, Linear, BatchNorm1d

from recommender.engine.pre_agent.preprocessing import USERS, SERVICES

SERVICES_AUTOENCODER = "Services Auto-Encoder mMdel"
USERS_AUTOENCODER = "Users Auto-Encoder Model"


class AutoEncoder(Module):
    """An Autoencoder model"""

    def __init__(self, features_dim, embedding_dim):
        super().__init__()

        self.encoder = Sequential(
            Linear(features_dim, 128),
            BatchNorm1d(128),
            Linear(128, 64),
            BatchNorm1d(64),
            Linear(64, embedding_dim),
            BatchNorm1d(embedding_dim),
        )

        self.decoder = Sequential(
            Linear(embedding_dim, 64),
            BatchNorm1d(64),
            Linear(64, 128),
            BatchNorm1d(128),
            Linear(128, features_dim),
        )

    def forward(self, features):
        embedding = self.encoder(features)
        reconstruction = self.decoder(embedding)

        return reconstruction


class UserAutoEncoder(AutoEncoder):
    """Autoencoder for Users"""


class ServiceAutoEncoder(AutoEncoder):
    """Autoencoder for Services"""


def create_autoencoder_model(
    collection_name,
    features_dim,
    embedding_dim,
    writer=None,
    train_ds_dl=None,
    device=torch.device("cpu"),
):
    """This function should be used to instantiate and Autoencoder rather
    than direct autoencoder class"""

    if collection_name == USERS:
        model = UserAutoEncoder(
            features_dim=features_dim, embedding_dim=embedding_dim
        ).to(device)
    elif collection_name == SERVICES:
        model = ServiceAutoEncoder(
            features_dim=features_dim, embedding_dim=embedding_dim
        ).to(device)
    else:
        raise ValueError

    if writer is not None and train_ds_dl is not None:
        batch = next(iter(train_ds_dl))
        example_input = batch[0].to(device)
        writer.add_graph(model, example_input)

    return model


def create_embedder(autoencoder):
    """It allows create Embedder model from the Autoencoder model"""

    embedder = deepcopy(autoencoder.encoder)
    for parameter in embedder.parameters():
        parameter.requires_grad = False

    return embedder
