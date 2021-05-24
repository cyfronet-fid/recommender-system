# pylint: disable=invalid-name, no-member, missing-module-docstring

from typing import List

import numpy as np
import torch
import pandas as pd

from recommender.engine.agents.rl_agent.utils import get_service_indices, iou
from recommender.models import User, Service


def _dist_score(x):
    """Returns score for a given euclidean distance, between 0 and 1.
    The lesser the distance the better the score"""
    return -x + 1


def approx_service_engagement(
    user: User,
    service: Service,
    engaged_services_history: List[Service],
    normalized_embedded_services: torch.Tensor,
    index_id_map: pd.DataFrame,
) -> float:
    """
    Approximates user's interest in the given service, based on common users
    and services categories and scientific_domains and
    distance from the centroid of user's engaged services
    embeddings and given service embedding.

    Args:
        user: user that we are trying to approximate engagement for
        service: service for which we are trying to approximate user's engagement
        engaged_services_history: list of user-engaged services
            to include in centroid distance score
        normalized_embedded_services: normalized service embeddings
        index_id_map: map that tells us which embedding indices
            are related to which db_ids

    Returns:
        user_engagement: user's interest in the given service
    """

    iou_categories = iou(set(user.categories), set(service.categories))
    iou_scientific_domains = iou(
        set(user.scientific_domains), set(service.scientific_domains)
    )

    if len(engaged_services_history) == 0:
        return np.array([iou_categories, iou_scientific_domains]).mean()

    engaged_service_ids = list(map(lambda s: s.id, engaged_services_history))

    embedded_engaged_services = normalized_embedded_services[
        get_service_indices(index_id_map, engaged_service_ids)
    ]
    embedded_service = normalized_embedded_services[
        get_service_indices(index_id_map, [service.id])
    ]

    engaged_services_centroid = embedded_engaged_services.mean(dim=0)
    dist = torch.cdist(embedded_service, engaged_services_centroid.view(1, -1)).view(-1)

    engaged_services_score = _dist_score(dist.item())

    return np.array(
        [iou_categories, iou_scientific_domains, engaged_services_score]
    ).mean()
