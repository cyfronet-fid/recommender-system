# pylint: disable=no-member, invalid-name, missing-module-docstring, too-many-locals
import random
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch

from recommender.engine.agents.rl_agent.reward_mapping import (
    TRANSITION_REWARDS_CSV_PATH,
)
from recommender.engine.agents.rl_agent.utils import use_service_embedder
from recommender.engine.models.autoencoders import create_embedder, SERVICES_AUTOENCODER
from recommender.engine.utils import load_last_module
from recommender.models import Service, State, SearchData, Sars, User
from recommender.services.fts import retrieve_services_for_recommendation
from recommender.services.synthetic_dataset.rewards import synthesize_reward
from recommender.services.synthetic_dataset.service_engagement import (
    approx_service_engagement,
)
from recommender.services.synthetic_dataset.users import synthesize_users


def _normalize_embedded_services(embedded_services: torch.Tensor) -> torch.Tensor:
    """Normalizes service embeddings that the max distance
    between any given 2 services is at most 1."""
    normalization_factor = (
        2 * torch.cdist(embedded_services, torch.zeros_like(embedded_services)).max()
    )
    output = embedded_services / normalization_factor
    return output


def _get_engaged_services(rewards_dict: Dict[Service, List[str]]) -> List[Service]:
    """Returns list of services that were engaged
    by user in some way, based on reward history"""
    return [service for service, rewards in rewards_dict.items() if len(rewards) > 1]


def _get_ordered_services(rewards_dict: Dict[Service, List[str]]) -> List[Service]:
    """Returns list of services that were ordered based on reward history"""
    return [
        service
        for service, rewards in rewards_dict.items()
        if len(rewards) > 1 and rewards[-1] == "order"
    ]


def _get_relevant_search_data(user, ordered_services, k):
    for c in user.categories:
        search_data = SearchData(categories=[c])
        if len(retrieve_services_for_recommendation(search_data, ordered_services)) > k:
            return search_data.save()

    for sd in user.scientific_domains:
        search_data = SearchData(scientific_domains=[sd])
        if len(retrieve_services_for_recommendation(search_data, ordered_services)) > k:
            return search_data.save()

    return SearchData().save()


def generate_dataset(
    user_samples: int,
    K: int = 3,
    interactions_range: Tuple[int, int] = (3, 10),
    cluster_distributions: Tuple[(float,) * 7] = (
        0.12,
        0.2,
        0.2,
        0.12,
        0.12,
        0.12,
        0.12,
    ),  # experimentally chosen
) -> Tuple[List[Sars], List[User]]:
    """
    Generates artificial SARS dataset that will be used for training
    and benchmarking the RL agent.
    It generates users belonging to a predefined category and scientific domain clusters
    and for each one generates specified number of SARSes by:
        - choosing an action randomly (LIRD is a off-policy RL algorithm)
        - approximating heuristically the engagement of a user in a given service
        - generates rewards based on this approximation for each service
            (simulates the page transitions in the Marketplace portal)

    Args:
        user_samples: number of users to generate
        K: number of recommended services in a single recommendation
        interactions_range: range of recommendations
            (as well as SARSes generated for each user)
        cluster_distributions: defines the distribution of user clusters

    Returns:
        Generated SARSes and synthetic users
    """

    synthesized_users = synthesize_users(
        user_samples, cluster_distributions=cluster_distributions
    )

    service_embedded_tensors, index_id_map = use_service_embedder(
        Service.objects.order_by("id"),
        create_embedder(load_last_module(SERVICES_AUTOENCODER)),
    )

    normalized_services = _normalize_embedded_services(service_embedded_tensors)
    transition_rewards_df = pd.read_csv(TRANSITION_REWARDS_CSV_PATH, index_col="source")
    sarses = []

    for user in synthesized_users:
        engaged_services = []
        ordered_services = []

        state = State(
            user=user,
            services_history=[],
            search_data=_get_relevant_search_data(user, ordered_services, K),
            synthetic=True,
        ).save()

        for _ in range(
            np.random.randint(interactions_range[0], interactions_range[1] + 1)
        ):
            # Here we could even use PreAgent or even RLAgent if it's trained
            action = random.sample(
                list(
                    retrieve_services_for_recommendation(
                        state.search_data, ordered_services
                    )
                ),
                k=K,
            )

            service_engagements = {
                s: approx_service_engagement(
                    user, s, engaged_services, normalized_services, index_id_map
                )
                for s in action
            }

            rewards_dict = {
                s: synthesize_reward(transition_rewards_df, engagement)
                for s, engagement in service_engagements.items()
            }

            engaged_services += _get_engaged_services(rewards_dict)
            ordered_services += _get_ordered_services(rewards_dict)

            next_state = State(
                user=user,
                services_history=engaged_services,
                search_data=_get_relevant_search_data(user, ordered_services, K),
                synthetic=True,
            ).save()

            sars = Sars(
                state=state,
                action=action,
                reward=list(rewards_dict.values()),
                next_state=next_state,
                synthetic=True,
            ).save()

            sarses.append(sars)
            state = next_state

    return sarses, synthesized_users