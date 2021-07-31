# pylint: disable=invalid-name, no-member, missing-module-docstring

from typing import List

import numpy as np
import torch
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

from definitions import LOG_DIR
from recommender.engine.agents.rl_agent.utils import get_service_indices, cfr
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

    common_cat_no = len(set(user.categories) & set(service.categories))
    common_sd_no = len(set(user.scientific_domains) & set(service.scientific_domains))

    x = common_cat_no + common_sd_no

    # Sigmoid function over threshold
    if x == 0:
        common_measure = 0
    else:
        common_measure = 1 / (1 + np.exp(-(x - 1)))

    # return common_measure

    if len(engaged_services_history) == 0:
        return common_measure

    engaged_service_ids = [s.id for s in engaged_services_history]

    embedded_engaged_services = normalized_embedded_services[
        get_service_indices(index_id_map, engaged_service_ids)
    ]
    embedded_service = normalized_embedded_services[
        get_service_indices(index_id_map, [service.id])
    ]

    engaged_services_centroid = embedded_engaged_services.mean(dim=0)
    dist = torch.cdist(embedded_service, engaged_services_centroid.view(1, -1)).view(-1)

    engaged_services_score = _dist_score(dist.item())

    user_engagement = np.array(
        [common_measure, engaged_services_score]
    ).mean()

    return user_engagement
