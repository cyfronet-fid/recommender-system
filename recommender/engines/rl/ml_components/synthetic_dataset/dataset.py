# pylint: disable=no-member, invalid-name, missing-module-docstring, too-many-locals, too-many-arguments
import random
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch

from recommender.engines.rl.ml_components.reward_mapping import (
    TRANSITION_REWARDS_CSV_PATH,
)
from recommender.engines.autoencoders.ml_components.embedder import Embedder
from recommender.engines.autoencoders.ml_components.normalizer import (
    Normalizer,
    NormalizationMode,
)
from recommender.models import Service, State, SearchData, Sars, User
from recommender.services.fts import retrieve_services_for_recommendation
from recommender.engines.rl.ml_components.synthetic_dataset.rewards import (
    synthesize_reward,
    RewardGeneration,
)
from recommender.engines.rl.ml_components.synthetic_dataset.service_engagement import (
    approx_service_engagement,
)
from recommender.engines.rl.ml_components.synthetic_dataset.users import (
    synthesize_users,
)


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
    return [service for service, rewards in rewards_dict.items() if len(rewards) > 0]


def _get_ordered_services(rewards_dict: Dict[Service, List[str]]) -> List[Service]:
    """Returns list of services that were ordered based on reward history"""
    return [
        service
        for service, rewards in rewards_dict.items()
        if len(rewards) > 0 and rewards[-1] == "order"
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
    service_embedder: Embedder,
    K: int = 3,
    interactions_range: Tuple[int, int] = (3, 10),
    reward_generation_mode: RewardGeneration = RewardGeneration.COMPLEX,
    simple_reward_threshold: int = 0.5,
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
        reward_generation_mode: either "complex" or "simple". Specifies if
            the simulated reward should be "complex" (drawn from the binomial
            distribution + simulating user MP transitions) or "simple"
            (high reward above a threshold - see "simple_reward_threshold" parameter)
        simple_reward_threshold: defines the threshold above which
            the simulated reward is high
        cluster_distributions: defines the distribution of user clusters

    Returns:
        Generated SARSes and synthetic users
    """

    synthesized_users = synthesize_users(
        user_samples, cluster_distributions=cluster_distributions
    )

    all_services = Service.objects.order_by("id")
    service_embedded_tensors, index_id_map = service_embedder(all_services)

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
                s: synthesize_reward(
                    transition_rewards_df,
                    engagement,
                    mode=reward_generation_mode,
                    simple_mode_threshold=simple_reward_threshold,
                )
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


def generate_synthetic_sarses(
    service_embedder: Embedder,
    K: List[int] = 3,
    interactions_range: Tuple[int, int] = (3, 10),
    reward_generation_mode: RewardGeneration = RewardGeneration.COMPLEX,
    simple_reward_threshold: int = 0.5,
) -> Tuple[List[Sars]]:
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
        K: number of recommended services in a single recommendation
        interactions_range: range of recommendations
            (as well as SARSes generated for each user)
        reward_generation_mode: either "complex" or "simple". Specifies if
            the simulated reward should be "complex" (drawn from the binomial
            distribution + simulating user MP transitions) or "simple"
            (high reward above a threshold - see "simple_reward_threshold" parameter)
        simple_reward_threshold: defines the threshold above which
            the simulated reward is high

    Returns:
        Generated SARSes and synthetic users
    """

    users = User.objects

    normalized_services, index_id_map = _embed_and_normalize(service_embedder)
    transition_rewards_df = pd.read_csv(TRANSITION_REWARDS_CSV_PATH, index_col="source")
    sarses = []

    for user in users:
        engaged_services = []
        ordered_services = []

        state = State(
            user=user,
            services_history=[],
            search_data=SearchData().save(),
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
                s: synthesize_reward(
                    transition_rewards_df,
                    engagement,
                    mode=reward_generation_mode,
                    simple_mode_threshold=simple_reward_threshold,
                )
                for s, engagement in service_engagements.items()
            }

            engaged_services += _get_engaged_services(rewards_dict)
            ordered_services += _get_ordered_services(rewards_dict)

            next_state = State(
                user=user,
                services_history=engaged_services,
                search_data=SearchData().save(),
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

    return sarses


def _embed_and_normalize(service_embedder):
    service_embedded_tensors, index_id_map = service_embedder(
        Service.objects.order_by("id"), use_cache=False, save_cache=False
    )
    normalizer = Normalizer(mode=NormalizationMode.NORM_WISE)
    normalized_services, _ = normalizer(service_embedded_tensors)

    return normalized_services, index_id_map
