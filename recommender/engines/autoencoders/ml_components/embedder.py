# pylint: disable=too-few-public-methods, fixme

"""Embedder"""

from copy import deepcopy
from typing import Iterable, Union

import pandas as pd
import torch
from mongoengine import QuerySet
from tqdm.auto import tqdm

from recommender.engines.autoencoders.ml_components.autoencoder import AutoEncoder
from recommender.engines.rl.utils import (
    create_index_id_map,
)
from recommender.engines.persistent_mixin import Persistent
from recommender.errors import MissingOneHotTensorError, MissingDenseTensorError
from recommender.models import User, Service

USER_EMBEDDER = "USER_EMBEDDER"
SERVICE_EMBEDDER = "SERVICE_EMBEDDER"


class Embedder(Persistent):
    """User or Services Embedder"""

    def __init__(self, autoencoder: AutoEncoder):
        self.disable_tqdm = True
        self.network = deepcopy(autoencoder.encoder)
        # self.network.eval() # TODO: find out why this breaks the embedder test
        self.one_hot_dim = self.network[0].in_features
        self.dense_dim = self.network[-1].out_features

        for parameter in self.network.parameters():
            parameter.requires_grad = False

    def __call__(
        self,
        objects: Union[Iterable[Union[User, Service]], QuerySet],
        use_cache: bool = True,
        save_cache: bool = False,
    ) -> (torch.Tensor, pd.DataFrame):
        """Embed objects one hot tensors into dense tensors.

        Args:
            objects: Iterable of objects that have one_hot_tensor and
             dense_tensor fields
            use_cache: If True and all dense tensors exist, they will be
             returned in a batch
            save_cache: Flag deciding whether to save dense tensors into
             objects.
        Returns:
            dense_tensors_batch: Batch of objects' dense tensors.
            index_id_map: Pandas Dataframe with index to id mapping.
        """
        objects = list(objects)
        index_id_map = create_index_id_map(objects)

        if self.dense_tensors_exist(objects):
            if use_cache:
                dense_tensors = [obj.dense_tensor for obj in objects]
                dense_tensors_batch = torch.Tensor(dense_tensors)
                return dense_tensors_batch, index_id_map
        else:
            if use_cache:
                raise MissingDenseTensorError

        if not self.one_hot_tensors_exist(objects):
            raise MissingOneHotTensorError

        one_hot_tensors = [obj.one_hot_tensor for obj in objects]
        one_hot_tensors_batch = torch.Tensor(one_hot_tensors)

        dense_tensors_batch = self.network(one_hot_tensors_batch)

        if save_cache:
            for obj, dense_tensor in tqdm(
                zip(objects, dense_tensors_batch),
                total=len(objects),
                disable=self.disable_tqdm,
            ):
                obj.dense_tensor = dense_tensor.tolist()
                obj.save()

        return dense_tensors_batch, index_id_map

    @staticmethod
    def dense_tensors_exist(objects: Union[Iterable, QuerySet]):
        """Check if embedded tensors exists

        Args:
            objects: list of users or services

        Return:
            True if embedded tensors of all objects exists, otherwise False
        """
        objects = list(objects)
        first_len = len(objects[0].dense_tensor)
        if first_len == 0:
            return False
        return all(len(obj.dense_tensor) == first_len for obj in objects)

    @staticmethod
    def one_hot_tensors_exist(objects: Union[Iterable, QuerySet]):
        """Check if one hot tensors exists
        Args:
            objects: list of users or services

        Return:
            True if one hot tensors of all objects exists, otherwise False
        """
        objects = list(objects)
        first_len = len(objects[0].one_hot_tensor)
        if first_len == 0:
            return False
        return all(len(obj.one_hot_tensor) == first_len for obj in objects)
