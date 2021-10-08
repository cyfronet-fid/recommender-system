# pylint: disable=too-few-public-methods

"""Normalizer"""

from typing import Union, Iterable

import torch
from mongoengine import QuerySet
from tqdm.auto import tqdm

from recommender.engine.agents.rl_agent.utils import dense_tensors_exist
from recommender.errors import (
    DifferentTypeObjectsInCollectionError,
    MissingDenseTensorError,
)
from recommender.models import User, Service


class Normalizer:
    """User or Services Normalizer"""

    def __init__(self):
        self.disable_tqdm = True

    def __call__(
        self,
        objects: Union[Iterable[Union[User, Service]], QuerySet],
        save_cache: bool = True,
        dimensions_maxes: Union[None, torch.Tensor] = None,
    ):
        """For the given objects calculate normalized dense tensors.

        Batch of objects' tensors can be viewed as a matrix where rows
         correspond to objects tensors and columns correspond to dimensions
         of these tensors. Each value in such a column is divided by the max
         value in this column.

        Args:
            objects: Services or users (without mixing)
            save_cache: Flag deciding if calculated normalised tensors should
             be saved into given objects.

        Returns:
            normalised_dense_tensors_batch: Batch of objects' normalized tensors.
            dimensions_maxes: Normalization factors per dimension
        """

        objects = list(objects)

        if not all(isinstance(obj, objects[0].__class__) for obj in objects):
            raise DifferentTypeObjectsInCollectionError

        if not dense_tensors_exist(objects):
            raise MissingDenseTensorError

        dense_tensors = [obj.dense_tensor for obj in objects]
        dense_tensors_batch = torch.Tensor(dense_tensors)

        if dimensions_maxes is None:
            dimensions_maxes = torch.max(torch.abs(dense_tensors_batch), 0).values
        normalised_dense_tensors_batch = dense_tensors_batch / dimensions_maxes

        if save_cache:
            for obj, dense_tensor in tqdm(
                zip(objects, normalised_dense_tensors_batch),
                total=len(objects),
                disable=self.disable_tqdm,
            ):
                obj.dense_tensor = dense_tensor.tolist()
                obj.save()

        return normalised_dense_tensors_batch, dimensions_maxes
