# pylint: disable=no-member, invalid-name, missing-function-docstring

"""RL Agent Utilities"""

from typing import Tuple, Iterable, Union, List

import pandas as pd
import torch
import torch.nn
from graphviz import Digraph
from mongoengine import QuerySet

from recommender.models import State, SearchData, User, Service
from recommender.models import UserAction
from recommender.services.services_history_generator import generate_services_history


def _get_visit_ids(user_action: UserAction) -> Tuple[str, str]:
    """
    Get source and target visit ids of the user action.

    Args:
        user_action: User action object.
    Returns:
        visit_ids: Tuple of the source and target visit ids.
    """

    rua_svid = user_action.source.visit_id
    rua_tvid = user_action.target.visit_id

    visit_ids = (str(rua_svid), str(rua_tvid))

    return visit_ids


def _generate_tree(user_action: UserAction, graph: Digraph) -> Digraph:
    """
    Utility function implementing recursive user actions tree generation.

    Args:
        user_action: User action that is the root of the tree.
        graph: Graph accumulator.
    """

    ua_svid, ua_tvid = _get_visit_ids(user_action)

    graph.node(ua_svid, label=ua_svid[:3])
    graph.node(ua_tvid, label=ua_tvid[:3])
    if user_action.action.order:
        color = "red"
        label = "Order"
    else:
        color = "black"
        label = ""

    if user_action.source.root is not None:
        service_id = user_action.source.root.service.id
        label = f"service id: {service_id}"
        color = "green"

    graph.edge(ua_svid, ua_tvid, color=color, label=label)

    children = list(UserAction.objects(source__visit_id=ua_tvid))
    for ua in children:
        _generate_tree(ua, graph)

    return graph


def generate_tree(user_action: UserAction) -> Digraph:
    """
    This method is used for generating the user actions' tree
    rooted in the user action given as a parameter.

    Args:
        user_action: User action that is the root of the tree.
    """

    graph = Digraph(comment="User Actions Tree")
    return _generate_tree(user_action, graph)


def create_index_id_map(services: Union[Iterable, QuerySet]):
    return pd.DataFrame([s.id for s in services], columns=["id"])


def create_itemspace(embedder: torch.nn.Module) -> Tuple[torch.Tensor, pd.DataFrame]:
    all_services = list(Service.objects.order_by("id"))
    service_embedded_tensors, index_id_map = use_service_embedder(
        all_services, embedder
    )

    return service_embedded_tensors, index_id_map


def use_service_embedder(
    services: Union[Iterable, QuerySet], embedder: torch.nn.Module
) -> Tuple[torch.Tensor, pd.DataFrame]:
    """
    Embed list of services with provided embedder.

    Args:
        services: List of services.
        embedder: Embeder model.

    Returns:
        embedded_services: embedded services.
        index_id_map: index to id mapping.
    """

    one_hot_service_tensors = torch.Tensor([s.tensor for s in services])

    with torch.no_grad():
        embedded_services = embedder(one_hot_service_tensors)

    index_id_map = create_index_id_map(services)

    return embedded_services, index_id_map


def get_service_indices(index_id_map: pd.DataFrame, ids: List[int]) -> List[int]:
    """Given a mapping between indices in the embedding and
    database ids returns indices of services with given ids."""

    id_index_map = pd.Series(index_id_map["id"].index.values, index=index_id_map["id"])

    # Below intersection is necessary to avoid key error in pandas .loc[]
    # Checking for existence in set is O(1) because it's hash-based,
    # so the overall complexity is O(len(ids))
    possible_values = set(id_index_map.index.values.tolist())
    valid_ids = [x for x in ids if x in possible_values]

    indices = id_index_map.loc[valid_ids].values.reshape(-1).tolist()

    return indices


def iou(set1: set, set2: set) -> float:
    """Get intersection over union factor of two sets"""
    return len(set1 & set2) / len(set1 | set2)


def create_state(user: User, search_data: SearchData) -> State:
    """
    Get needed information from context and create state.

    Args:
        user: MongoEngine User object.
        search_data: SearchData object.

    Returns:
        state: State containing information about user and search_data
    """

    state = State(
        user=user,
        services_history=generate_services_history(user),
        search_data=search_data,
    )

    return state


def use_user_embedder(
    users: List[User], user_embedder: torch.nn.Module
) -> torch.Tensor:
    """
    # User cannot be directly embedded as a single example batch because it
    #  is not handled by torch BatchNorm1D. To embedd it successfully it has
    #  to be multiplied and embedded after it. Finally only one embedded
    #  tensor is returned.
    #
    # There is a "safe_batch_size" constant that has been arbitrarily set to
    #  64 (there are some proofs that batchnorm can perform poorly if
    #  batch_size is small (<32). It is probably important during training
    #  - not inference - but this part of code will need some attention
    #  during polishing this project. It is possible that safe_batch_size
    #  should be same as batch_size during training. So, to sum it up, it's a
    #   TODO)
    # TODO remove above bullshit XD

    Args:
        users: List of MongoEngine User objects.
        user_embedder: User Embedder model.

    Returns:
        embedded_user_tensor: Embedded user tensors
    """

    user_tensors_batch = torch.Tensor([user.tensor for user in users])

    with torch.no_grad():
        embedded_user_tensors_batch = user_embedder(user_tensors_batch)

    return embedded_user_tensors_batch
