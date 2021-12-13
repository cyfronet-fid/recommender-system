# pylint: disable=missing-module-docstring, missing-class-docstring, invalid-name, too-few-public-methods, fixme

"""Normalizer"""
from enum import auto, Enum
from typing import Union, Iterable

import torch
from mongoengine import QuerySet
from tqdm.auto import tqdm

from recommender.engines.autoencoders.ml_components.embedder import Embedder
from recommender.errors import (
    DifferentTypeObjectsInCollectionError,
    MissingDenseTensorError,
)
from recommender.models import User, Service
from logger_config import get_logger

logger = get_logger(__name__)


class NormalizationMode(Enum):
    DIMENSION_WISE = auto()
    NORM_WISE = auto()


class Normalizer:
    """User or Services Normalizer"""

    def __init__(self, mode: NormalizationMode = NormalizationMode.DIMENSION_WISE):
        self.disable_tqdm = True
        self.mode = mode

    def __call__(
        self,
        objects: Union[Iterable[Union[User, Service]], QuerySet, torch.Tensor],
        save_cache: bool = False,
        normalization_factors: Union[None, torch.Tensor] = None,
    ):
        """For the given objects calculate normalized_tensors dense tensors.

        Batch of objects' tensors can be viewed as a matrix where rows
         correspond to objects tensors and columns correspond to dimensions
         of these tensors. Each value in such a column is divided by the max
         value in this column.

        Args:
            objects: Services or users (without mixing) or a batch of tensors.
            save_cache: Flag deciding if calculated normalised tensors should
             be saved into given objects. Ignored if called for batch of tensors.

        Returns:
            normalized_batch: Batch of objects' normalized_tensors tensors.
            normalization_factors: Normalization factors per dimension
        """

        if isinstance(objects, torch.Tensor):
            return self._normalize_batch(objects, normalization_factors)

        return self._normalize_objects(objects, save_cache, normalization_factors)

    def _normalize_batch(self, batch, normalization_factors):
        if normalization_factors is None:
            if self.mode == NormalizationMode.DIMENSION_WISE:
                normalization_factors = self._dimension_wise_factors(batch)
            else:
                normalization_factors = self._norm_wise_factors(batch)

        normalized_batch = batch / normalization_factors
        return normalized_batch, normalization_factors

    def _normalize_objects(self, objects, save_cache, normalization_factors):
        objects = list(objects)

        if not all(isinstance(obj, objects[0].__class__) for obj in objects):
            raise DifferentTypeObjectsInCollectionError()

        if not Embedder.dense_tensors_exist(objects):
            raise MissingDenseTensorError()

        tensors = [obj.dense_tensor for obj in objects]
        batch = torch.Tensor(tensors)
        normalized_batch, normalization_factors = self._normalize_batch(
            batch, normalization_factors
        )

        if save_cache:
            for obj, dense_tensor in tqdm(
                zip(objects, normalized_batch),
                total=len(objects),
                disable=self.disable_tqdm,
            ):
                obj.dense_tensor = dense_tensor.tolist()
                obj.save()

        return normalized_batch, normalization_factors

    @staticmethod
    def _dimension_wise_factors(batch):
        return torch.max(torch.abs(batch), dim=0).values

    @staticmethod
    def _norm_wise_factors(batch):
        return batch.norm(dim=1).max().broadcast_to(batch.shape[1])
