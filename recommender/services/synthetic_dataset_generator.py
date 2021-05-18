# pylint: disable=no-member, missing-function-docstring, disable=missing-module-docstring, invalid-name
from typing import List

import numpy as np
import torch
import pandas as pd

from recommender.engine.agents.rl_agent.utils import get_service_indices, iou
from recommender.models import User, Service
from recommender.services.services_history_generator import get_ordered_services


def _normalize_embedded_services(embedded_services: torch.Tensor) -> torch.Tensor:
    """Normalizes service embeddings that the max distance
    between any given 2 services is at most 1."""
    normalization_factor = (
        2 * torch.cdist(embedded_services, torch.zeros_like(embedded_services)).max()
    )
    output = embedded_services / normalization_factor
    return output


def _dist_score(x):
    """Returns score for a given euclidean distance, between 0 and 1.
    The lesser the distance the better the score"""
    return -x + 1


def approx_service_engagement(
    user: User,
    service: Service,
    normalized_embedded_services: torch.Tensor,
    index_id_map: pd.DataFrame,
) -> float:
    """
    Approximates user's interest in the given service, based on common users
    and services categories and scientific_domains and
    distance from the centroid of user's ordered_services
    embeddings and given service embedding.

    Args:
        user: user that we are trying to approximate engagement for
        service: service for which we are trying to approximate user's engagement
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

    ordered_services = get_ordered_services(user)

    if len(ordered_services) == 0:
        return np.array([iou_categories, iou_scientific_domains]).mean()

    ordered_service_ids = list(map(lambda s: s.id, ordered_services))

    embedded_ordered_services = normalized_embedded_services[
        get_service_indices(index_id_map, ordered_service_ids)
    ]
    embedded_service = normalized_embedded_services[
        get_service_indices(index_id_map, [service.id])
    ]

    embedded_accessed_services_centroid = embedded_ordered_services.mean(dim=0)
    dist = torch.cdist(
        embedded_service, embedded_accessed_services_centroid.view(1, -1)
    ).view(-1)

    ordered_services_score = _dist_score(dist.item())

    return np.array(
        [iou_categories, iou_scientific_domains, ordered_services_score]
    ).mean()


# REWARDS that are listed in increasing interest order
REWARDS = ["exit", "simple_transition", "mild_interest", "interest", "order"]


def _draw_reward(engagement):
    """Draws reward based on engagement. It is based on binomial distribution.
    The intuition is that the higher the engagement,
    the easier it is for the distribution to pick
    high-engagement action like interest or reward and vice-versa."""
    reward = np.random.binomial(len(REWARDS) - 1, engagement)
    return REWARDS[reward]


def _get_closest_reward(required_reward, available_rewards):
    """Returns reward closest to the required reward in terms of engagement"""
    for r in reversed(REWARDS[: REWARDS.index(required_reward) + 1]):
        if r in available_rewards:
            return r
    return "exit"


def construct_reward(
    transitions_df: pd.DataFrame,
    engagement: float,
    max_depth: int = 10,
    current_depth: int = 0,
    source: str = "/services",
) -> List[str]:
    """
    Synthesises reward using service_engagement factor.
    It recursively wanders around the transition graph,
    and chooses appropriate rewards based on engagement
    and binomial distribution of engagement over rewards.

    Args:
        transitions_df: transition adjacency matrix, with reward mapping
        engagement: service engagement that is returned by
            approx_service_engagement (between 0 and 1)
        max_depth: max depth of the recursive stack
        current_depth: current depth of the stack, should always be 0 on the first call
        source: starting page, should always equal to /services on the first call

    Returns:
        reward list: list of symbolic rewards given for a given graph walk
    """

    if current_depth >= max_depth:
        return []

    available_targets = transitions_df.loc[source, :]
    known_targets = available_targets[
        transitions_df.loc[source, :].keys().drop("unknown_page_id")
    ]
    valid_targets = known_targets[known_targets != "unknown_transition"]

    drawn_reward = _get_closest_reward(
        _draw_reward(engagement), valid_targets.values.tolist()
    )

    if drawn_reward == "order":
        return [drawn_reward]
    if drawn_reward == "exit":
        return []

    drawn_target = np.random.choice(valid_targets[valid_targets == drawn_reward].keys())

    return [drawn_reward] + construct_reward(
        transitions_df, engagement, max_depth, current_depth + 1, source=drawn_target
    )
