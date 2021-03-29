# pylint: disable=no-member

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
from typing import List

from recommender.models import UserAction, Recommendation, State, Sars, Service
from recommender.engine.reward_mapping import ua_to_reward_id


RECOMMENDATION_PAGES_IDS = ("catalogue_services_list",)


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
    """Find all user's root actions that user took before
    current recommendation"""

    if recommendation.user:
        root_uas_before = root_uas(
            user=recommendation.user, timestamp__lt=recommendation.timestamp
        )
    else:
        root_uas_before = root_uas(
            unique_id=recommendation.unique_id,
            timestamp__lt=recommendation.timestamp,
        )

    return root_uas_before.order_by("+timestamp")


def _ruas2services(root_uas):
    """This function maps root user actions to services"""

    return [root_ua.source.root.service for root_ua in root_uas]


def generate_sarses():
    """Use this method to generate SARSes in the database"""

    # Find all root user actions rooted in recommendation panel
    root_uas = UserAction.objects(source__root__type__="recommendation_panel")

    for recommendation in Recommendation.objects:
        # Create reward
        clicked_services_after, reward = _get_clicked_services_and_reward(
            recommendation, root_uas
        )

        # Create state
        root_uas_before = _find_root_uas_before(root_uas, recommendation)
        clicked_services_before = _ruas2services(root_uas_before)
        services_history_before = (
            recommendation.user.accessed_services + clicked_services_before
        )
        # Make unique but preserve order
        services_history_before = list(dict.fromkeys(services_history_before))
        state = State(
            user=recommendation.user,
            services_history=services_history_before,
            last_search_data=recommendation.search_data,
        )

        # create action
        action = recommendation.services
        # Create next state
        services_history_after = services_history_before + clicked_services_after
        # Make unique but preserve order
        services_history_after = list(dict.fromkeys(services_history_after))
        next_state = State(
            user=recommendation.user,
            services_history=services_history_after,
            last_search_data=recommendation.search_data,
        )

        # Create SARS
        Sars(state=state, action=action, reward=reward, next_state=next_state).save()

    return Sars.objects
