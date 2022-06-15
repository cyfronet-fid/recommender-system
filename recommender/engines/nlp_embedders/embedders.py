from typing import Union, Iterable, List

import torch
from mongoengine import QuerySet
from torch import Tensor

from recommender.engines.nlp_embedders.handlers import (
    Handler,
    ListOfObjectsHandler,
    StringHandler,
    ListOfStringsHandler,
    ObjectHandler,
)

SERVICE_EMBEDDING_DIM = "SE"
USER_EMBEDDING_DIM = "UE"


class Objects2tensorsEmbedder:
    def __init__(self, handlers: List[Handler]):
        self.handlers = handlers

    def __call__(
        self, objects: Union[Iterable, QuerySet], concat: bool = True
    ) -> Union[Tensor, List[Tensor]]:
        """
        Transform given objects into batch (or batches) of tensors.

        Args:
            objects: Iterable of mongoengine models: users or services.
            handlers: List of Handlers for fields that should be taken into consideration.
            concat: If True return batch of concatenated tensors of all fields and subfields of objects, if False return separate batches for all fields and subfields of objects.

        Returns:
            Batch or batches of tensors.
        """

        [handler(obj) for obj in objects for handler in self.handlers]
        objects_tensors = sum([handler.get_results() for handler in self.handlers], [])
        if concat:
            objects_tensor = torch.cat(objects_tensors, 1)
            # return objects_tensor
            return objects_tensor[
                :, :512
            ]  # TODO: this is hard cut, only for smaller size, it should be fixed before the merge
        else:
            return objects_tensors

    @property
    def embedding_dim(self):
        # return sum(handler.embedding_dim for handler in self.handlers)
        return 512  # TODO: this is hard cut, only for smaller size, it should be fixed before the merge


class Users2tensorsEmbedder(Objects2tensorsEmbedder):
    """Transform users into batch (or batches) of tensors."""

    def __init__(self):
        super().__init__(
            [
                ListOfObjectsHandler("categories", ["name"]),
                ListOfObjectsHandler("scientific_domains", ["name"]),
            ]
        )


class Services2tensorsEmbedder(Objects2tensorsEmbedder):
    """Transform services into batch (or batches) of tensors."""

    def __init__(self):
        super().__init__(
            [
                StringHandler("name"),
                StringHandler("description"),
                StringHandler("tagline"),
                ListOfStringsHandler("countries"),
                ListOfObjectsHandler("categories", ["name"]),
                ListOfObjectsHandler("providers", ["name"]),
                ObjectHandler("resource_organisation", ["name"]),
                ListOfObjectsHandler("scientific_domains", ["name"]),
                ListOfObjectsHandler("platforms", ["name"]),
                ListOfObjectsHandler("target_users", ["name", "description"]),
                ListOfObjectsHandler("access_modes", ["name", "description"]),
                ListOfObjectsHandler("access_types", ["name", "description"]),
                ListOfObjectsHandler("trls", ["name", "description"]),
                ListOfObjectsHandler("life_cycle_statuses", ["name", "description"]),
            ]
        )
