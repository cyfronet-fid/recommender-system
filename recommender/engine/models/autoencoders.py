# pylint: disable=no-member, too-many-arguments, missing-function-docstring

"""This module contains Autoencoders models instantiating-related functions"""

from copy import deepcopy

import torch
from torch.nn import Module, Sequential, Linear, BatchNorm1d, ReLU
from tqdm.auto import tqdm

from recommender.engine.preprocessing import USERS, SERVICES
from recommender.engine.utils import load_last_module
from recommender.models import User, Service

import torch.nn.functional as F

SERVICES_AUTOENCODER = "Services Auto-Encoder mMdel"
USERS_AUTOENCODER = "Users Auto-Encoder Model"


class AutoEncoder(Module):
    """An Autoencoder model"""

    def __init__(self, features_dim, embedding_dim):
        super().__init__()

        self.encoder = Sequential(
            Linear(features_dim, 128),
            BatchNorm1d(128),
            ReLU(),
            Linear(128, 64),
            BatchNorm1d(64),
            ReLU(),
            Linear(64, embedding_dim),
        )

        self.decoder = Sequential(
            Linear(embedding_dim, 64),
            BatchNorm1d(64),
            ReLU(),
            Linear(64, 128),
            BatchNorm1d(128),
            ReLU(),
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

def normalize(matrix: torch.Tensor) -> torch.Tensor:
    """Gets matrix that can be decomposed into row vectors and divide each
     dimension of these vectors by max value in this dimension (column)

     Args:
         matrix: input matrix of shape: [vectors_num, dimensions_num]
     Returns:
         normalised_matrix: normalised matrix of the same shape as input.
     """

    dimensions_maxes = torch.max(torch.abs(matrix), 0).values
    normalised_matrix = matrix / dimensions_maxes

    return normalised_matrix


def precalc_embedded_tensors(collection_name):
    if collection_name == USERS:
        embedder_name = USERS_AUTOENCODER
        Collection = User
    elif collection_name == SERVICES:
        embedder_name = SERVICES_AUTOENCODER
        Collection = Service
    else:
        raise Exception("Invalid collection name")

    embedder = create_embedder(load_last_module(embedder_name))

    ids = []
    tensors = []
    for obj in tqdm(Collection.objects, total=len(Collection.objects)):
        ids.append(obj.id)
        tensors.append(obj.tensor)

    embedded_tensors_batch = embedder(torch.Tensor(tensors))

    embedded_tensors_batch = normalize(embedded_tensors_batch)

    for id, embedded_tensor in tqdm(zip(ids, embedded_tensors_batch), total=len(ids)):
        obj = Collection.objects(id=id).first()
        obj.embedded_tensor = embedded_tensor.tolist()
        obj.save()
