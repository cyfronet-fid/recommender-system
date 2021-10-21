# pylint: disable=invalid-name, no-member, missing-module-docstring

from typing import List

import numpy as np
import torch
import pandas as pd

from recommender.engines.rl.utils import get_service_indices
from recommender.models import User, Service


def _distance_metric(cluster: torch.Tensor, point: torch.Tensor) -> float:
    """Computes euclidean distance between the cluster centroid and
    a given point and the converts it to a score, between 0 and 1,
    assuming the cluster and the point are normalized_tensors, with norm at max 1.0.
    The lesser the distance the better the score"""
    centroid = cluster.mean(dim=0)
    dist = (point - centroid).norm()
    # Scale the distance by 2, because the furthest
    # the points can be while normalized is 2.0
    dist /= 2
    # Map the distance so that the closer the points are,
    # the better the score
    return -dist + 1


def _overlap_metric(x: int) -> float:
    """Returns measure of overlapping niches.
    Uses a sigmoid function that is shifted to the right"""
    return 1 / (1 + np.exp(-(x - 1)))


def _compute_distance_score(
    engaged_services_history: List[Service],
    service: Service,
    normalized_embedded_services: torch.Tensor,
    index_id_map: pd.DataFrame,
):

    engaged_service_ids = [s.id for s in engaged_services_history]
    embedded_engaged_services = normalized_embedded_services[
        get_service_indices(index_id_map, engaged_service_ids)
    ]
    embedded_service = normalized_embedded_services[
        get_service_indices(index_id_map, [service.id])
    ][0]
    distance_score = _distance_metric(embedded_engaged_services, embedded_service)
    return distance_score


def _compute_overlap_score(user: User, service: Service):
    common_categories = set(user.categories) & set(service.categories)
    common_scientific_domains = set(user.scientific_domains) & set(
        service.scientific_domains
    )
    common_len = len(common_categories) + len(common_scientific_domains)
    overlap_score = _overlap_metric(common_len)
    return overlap_score


def approx_service_engagement(
    user: User,
    service: Service,
    engaged_services_history: List[Service],
    normalized_embedded_services: torch.Tensor,
    index_id_map: pd.DataFrame,
) -> float:
    """
    Approximates user's interest in the given service, based on overlapping users'
    and services' categories and scientific_domains, and the
    distance from the centroid of user's services_history
    embeddings and a given service embedding.

    Args:
        user: user that we are trying to approximate the engagement for
        service: service for which we are trying to approximate user's engagement
        engaged_services_history: list of user-engaged services
            to include in centroid distance score
        normalized_embedded_services: normalized_tensors service embeddings
        index_id_map: map that tells us which embedding indices
            are related to which db_ids

    Returns:
        user_engagement: user's interest in the given service
    """

    overlap_score = _compute_overlap_score(user, service)

    if len(engaged_services_history) == 0:
        return overlap_score

    distance_score = _compute_distance_score(
        engaged_services_history, service, normalized_embedded_services, index_id_map
    )

    user_engagement = float(np.mean([overlap_score, distance_score]))

    return user_engagement
