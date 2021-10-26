# pylint: disable=no-member, line-too-long, too-few-public-methods, fixme

"""
This module contains logic for composing marketplace DB dump,
recommendations and user actions into SARSes
needed for Recommender system's replay buffer.

marketplace DB dump ---\
                        \
recommendations ---------+----> SARSes
                        /
user actions ----------/

It assumes that above data are stored in database before call.
"""

import itertools
import multiprocessing
from typing import List

import torch

from recommender.models import User, UserAction, Recommendation, State, Sars, Service
from recommender.engines.rl.ml_components.reward_mapping import ua_to_reward_id
from recommender.engines.rl.ml_components.services_history_generator import (
    concat_histories,
)

RECOMMENDATION_PAGES_IDS = ("/services",)


def _tree_collapse(user_action):
    """Collapse tree of user actions rooted in user_action
    into list of reward ids"""

    # Accumulator
    rewards = [ua_to_reward_id(user_action)]

    # Recursion stop condition
    if user_action.action.order:
        return rewards

    if user_action.target.page_id in RECOMMENDATION_PAGES_IDS:
        return rewards

    target_id = user_action.target.visit_id
    children = list(UserAction.objects(source__visit_id=target_id))
    rewards += list(itertools.chain(*[_tree_collapse(child) for child in children]))

    return rewards


def _get_clicked_services_and_reward(
    recommendation, root_uas
) -> (List[Service], List[List[str]]):
    """Collapse root user actions trees to:
    - clicked_services_after
    - rewards lists (reward)"""

    reward = []
    clicked_services_after = []
    for service in recommendation.services:
        service_root_uas = root_uas(
            source__visit_id=recommendation.visit_id, source__root__service=service
        )
        if service_root_uas:
            clicked_services_after.append(service)
        service_rewards = list(
            itertools.chain(
                *[
                    _tree_collapse(service_root_ua)
                    for service_root_ua in service_root_uas
                ]
            )
        )
        reward.append(service_rewards)

    return clicked_services_after, reward


def _find_root_uas_before(root_uas, recommendation):
    """Find all user's root actions that has been taken before
    current recommendation"""

    if recommendation.user is not None:
        root_uas_for_user = root_uas(user=recommendation.user)
    else:
        root_uas_for_user = root_uas(unique_id=recommendation.unique_id)

    root_uas_before = root_uas_for_user(
        timestamp__lt=recommendation.timestamp
    ).order_by("+timestamp")

    return root_uas_before


def _get_empty_user() -> User:
    """
    This function get empty user with id=-1 from the database or create one.
    It fill its tensors with zeros to simulate tensor precalculation.
    It is used as a anonymous user and it has no categories or scientific
     domains.
    """

    assert len(User.objects) >= 1  # Necessary assumption

    true_user = User.objects.first()
    oht_shape = len(true_user.one_hot_tensor)
    dt_shape = len(true_user.dense_tensor)

    user = User.objects(id=-1).first()
    if user is None:
        user = User(
            id=-1,
            # Here is an assumption that anonymous user should be empty and
            # therefore its tensors should be zeros.
            # TODO: maybe Embedder should be used?
            one_hot_tensor=torch.zeros(oht_shape),
            dense_tensor=torch.zeros(dt_shape),
        )

    return user


def _get_next_recommendation(recommendation):
    """Given recommendation, this function determine if it's for logged or
    anonymous user and then it find the first next recommendation (in the
     chronological order) for this user.
    """

    if recommendation.user is not None:
        recommendations_for_user = Recommendation.objects(user=recommendation.user)
    else:
        recommendations_for_user = Recommendation.objects(
            unique_id=recommendation.unique_id
        )

    next_recommendation = (
        recommendations_for_user(timestamp__gt=recommendation.timestamp)
        .order_by("+timestamp")
        .first()
    )

    return next_recommendation


def generate_sars(recommendation, root_uas):
    """Generate sars for given recommendation and root user actions"""

    user = recommendation.user or _get_empty_user()

    # Create reward
    clicked_services_after, reward = _get_clicked_services_and_reward(
        recommendation, root_uas
    )

    # Create state
    root_uas_before = _find_root_uas_before(root_uas, recommendation)
    accessed_services = user.accessed_services
    services_history_before = concat_histories(accessed_services, root_uas_before)
    state = State(
        user=user,
        services_history=services_history_before,
        search_data=recommendation.search_data,
    ).save()

    # Create action
    action = recommendation.services

    # Create next state
    services_history_after = services_history_before + clicked_services_after

    next_recommendation = _get_next_recommendation(recommendation)
    if next_recommendation is None:
        return

    next_state = State(
        user=user,
        services_history=services_history_after,
        search_data=next_recommendation.search_data,
    ).save()

    # Create SARS
    Sars(state=state, action=action, reward=reward, next_state=next_state).save()


class Executor:
    """Executor needed for mapping sars generation over recommendations and
    processes pool"""

    def __init__(self, root_uas):
        self.root_uas = root_uas

    def __call__(self, recc):
        generate_sars(recc, self.root_uas)


def generate_sarses(multi_processing=True):
    """Use this method to generate SARSes in the database"""

    # Find all root user actions rooted in recommendation panel
    root_uas = UserAction.objects(source__root__type__="recommendation_panel")

    if multi_processing:
        executor = Executor(root_uas)
        cpu_n = multiprocessing.cpu_count()
        with multiprocessing.Pool(cpu_n) as pool:
            pool.map(executor, list(Recommendation.objects))
    else:
        for recommendation in Recommendation.objects:
            generate_sars(recommendation, root_uas)

    return Sars.objects
