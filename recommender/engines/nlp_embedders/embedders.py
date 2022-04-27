from typing import Union, Iterable, List

import torch
from mongoengine import QuerySet
from torch import Tensor

from recommender import User
from recommender.engines.nlp_embedders.handlers import (
    Handler,
    ListOfObjectsHandler,
    StringHandler,
    ListOfStringsHandler,
    ObjectHandler,
)
from recommender.models import Service


def objects2tensors(
    objects: Union[Iterable, QuerySet], handlers: List[Handler], concat: bool = True
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

    [handler(obj) for obj in objects for handler in handlers]
    objects_tensors = sum([handler.get_results() for handler in handlers], [])
    if concat:
        objects_tensor = torch.cat(objects_tensors, 1)
        return objects_tensor
    else:
        return objects_tensors

# from itertools import partial refactor
def users2tensor(
    users: Union[Iterable[User], QuerySet], concat: bool = True
) -> Union[Tensor, List[Tensor]]:
    """
    Transform users into batch (or batches) of tensors.

    Args:
        users: Mongoengine users.
        concat: If True returns batch of concatenated tensors of all fields and subfields of users, if False returns separate batches for all fields and subfields of users.

    Returns:
        Batch or batches of tensors.
    """

    handlers = [
        ListOfObjectsHandler("categories", ["name"]),
        ListOfObjectsHandler("scientific_domains", ["name"]),
    ]

    result = objects2tensors(users, handlers, concat)

    return result


def services2tensor(
    services: Union[Iterable[Service], QuerySet], concat: bool = True
) -> Union[Tensor, List[Tensor]]:
    """
    Transform services into batch (or batches) of tensors.

    Args:
        services: Mongoengine users.
        concat: If True return batch of concatenated tensors of all fields and subfields of services, if False return separate batches for all fields and subfields of services.

    Returns:
        Batch or batches of tensors.
    """

    handlers = [
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

    result = objects2tensors(services, handlers, concat)

    return result
