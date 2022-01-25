# pylint: disable=no-member, line-too-long, too-few-public-methods, invalid-name, fixme

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
from typing import List, Union

import torch
from mongoengine import DoesNotExist
from tqdm.auto import tqdm

from recommender.models import User, UserAction, Recommendation, State, Sars, Service
from recommender.engines.rl.ml_components.reward_mapping import ua_to_reward_id
from recommender.engines.rl.ml_components.services_history_generator import (
    concat_histories,
)
from logger_config import get_logger

RECOMMENDATION_PAGES_IDS = ("/services",)
logger = get_logger(__name__)


class Executor:
    """Executor needed for mapping sars generation over recommendations and
    processes pool"""

    def __init__(self, root_uas):
        self.root_uas = root_uas

    def __call__(self, recc):
        generate_sars(recc, self.root_uas)


def _tree_collapse(user_action, progress_bar=None):
    """Collapse tree of user actions rooted in user_action
    into list of reward ids"""
    if progress_bar is not None:
        progress_bar.update(1)

    # Accumulator
    rewards = [ua_to_reward_id(user_action)]

    # Recursion stop condition
    if user_action.action.order:
        return rewards

    if user_action.target.page_id in RECOMMENDATION_PAGES_IDS:
        return rewards

    target_id = user_action.target.visit_id
    children = list(UserAction.objects(source__visit_id=target_id))
    rewards += list(
        itertools.chain(*[_tree_collapse(child, progress_bar) for child in children])
    )

    return rewards


def _get_clicked_services_and_reward(
    recommendation, root_uas, progress_bar=None
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
                    _tree_collapse(service_root_ua, progress_bar)
                    for service_root_ua in service_root_uas
                ]
            )
        )
        reward.append(service_rewards)

    return clicked_services_after, reward


def _find_root_uas_before(root_uas, recommendation):
    """Find all user's root actions that have been taken before
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
    It fills its tensors with zeros to simulate tensor precalculation.
    It is used as an anonymous user, and it has no categories or scientific domains.
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
    anonymous user, and then it finds the first next recommendation (in the
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


def find_recs_based_on_uas(ua: UserAction) -> Union[Recommendation, None]:
    """Find user action's source recommendation."""

    while True:
        prev_ua = ua
        svid = prev_ua.source.visit_id
        recommendation = Recommendation.objects(visit_id=svid).first()
        if recommendation is not None:
            return recommendation
        ua = UserAction.objects(target__visit_id=svid).first()
        if ua is None:
            return None


def get_recs_for_update(verbose=True) -> List[Recommendation]:
    """Get recommendations that should be used to select SARSes for update"""

    # Get recommendations that are not processed so far
    # It takes care about recommendations that aren't root for any user actions
    if verbose:
        logger.info("Processing not processed recommendations...")
    recs_for_update = Recommendation.objects(processed__in=[False, None])
    recs_for_update_direct = set(recs_for_update)
    recs_for_update.update(processed=True)

    # Get recommendations based on user actions that are not processed so far
    # It takes care about recommendations that have the processed flag set to True, but
    # they need to be regenerated due to occurrence(s) of new user actions in the
    # user actions trees rooted in these recommendations.

    if verbose:
        logger.info("Processing not processed user actions...")
    not_processed_uas = UserAction.objects(processed__in=[False, None])
    not_processed_uas_direct = set(not_processed_uas)
    not_processed_uas.update(processed=True)

    if verbose:
        logger.info("Associating users actions with recommendations...")
    recs_for_update_from_uas = {
        find_recs_based_on_uas(ua)
        for ua in tqdm(
            not_processed_uas_direct, disable=not verbose, desc="User actions"
        )
    }

    if verbose:
        logger.info("Merging all gathered recommendation sets...")
    recs_for_update_from_uas = set(
        filter(lambda x: x is not None, recs_for_update_from_uas)
    )
    recs_for_update = list(recs_for_update_from_uas | recs_for_update_direct)

    if verbose:
        logger.info(
            "In general, there are %s recommendations based on which SARSes will be generated.",
            len(recs_for_update),
        )
        logger.info("Removing SARSes that should be regenerated...")

    Sars.objects(source_recommendation__in=([None] + recs_for_update)).delete()

    return recs_for_update


def missing_data_skipper(func):
    """This decorator makes `generate_sars` function able to gracefully skip missing data in the DB"""

    def skipable_func(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except DoesNotExist:
            pass

    return skipable_func


@missing_data_skipper
def generate_sars(recommendation, root_uas, progress_bar=None):
    """Generate sars for given recommendation and root user actions"""

    user = recommendation.user or _get_empty_user()

    # Create reward
    clicked_services_after, reward = _get_clicked_services_and_reward(
        recommendation, root_uas, progress_bar
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
    Sars(
        state=state,
        action=action,
        reward=reward,
        next_state=next_state,
        source_recommendation=recommendation,
    ).save()


def generate_sarses(
    multi_processing: bool = True,
    recommendations: Union[List[Recommendation], None] = None,
    verbose=True,
):
    """Use this method to generate SARSes in the database"""

    # Find all root user actions rooted in recommendation panel
    root_uas = UserAction.objects(source__root__type__="recommendation_panel")

    if recommendations is None:
        recommendations = list(Recommendation.objects)

    if multi_processing:
        executor = Executor(root_uas)
        cpu_n = multiprocessing.cpu_count()
        with multiprocessing.Pool(cpu_n) as pool:
            # list(tqdm(pool.imap(executor, recommendations), total=len(recommendations), disable=not verbose))
            pool.map(executor, recommendations)
    else:
        if verbose:
            logger.info("Regenerating SARS for each qualified recommendation...")
        uas_p_bar = tqdm(
            total=len(UserAction.objects),
            leave=True,
            disable=not verbose,
            desc="User Actions",
        )
        for recommendation in tqdm(
            recommendations,
            disable=not verbose,
            desc="Generating SARSes from recommendations",
        ):
            generate_sars(recommendation, root_uas, uas_p_bar)

    return Sars.objects


def regenerate_sarses(multi_processing=False, verbose=True):
    """Based on new user actions add new SARSes and regenerate existing ones that are deprecated"""

    # TODO: paralellization of get_recs_for_update
    recs_for_update = get_recs_for_update(verbose=True)

    sarses = generate_sarses(
        multi_processing=multi_processing,
        recommendations=recs_for_update,
        verbose=verbose,
    )

    return sarses
