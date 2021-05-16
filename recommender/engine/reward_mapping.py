"""This module contains logic needed for mapping user actions to rewards"""
import os

import pandas as pd

from definitions import ROOT_DIR
from recommender.models import UserAction

TRANSITION_REWARDS_CSV_PATH = os.path.join(
    ROOT_DIR, "resources", "transition_rewards.csv"
)


def _to_abstract_page_id(page_id: str):
    split = page_id.split("/")

    if len(split) < 2:
        return "unknown_page_id"
    if split[1] == "comparisons":
        return page_id
    if split[1] == "services" and len(split) > 2:
        split[2] = "{id}"
        return "/".join(split)

    return "unknown_page_id"


def ua_to_reward_id(user_action: UserAction) -> str:
    """
    This function should maps user_action to the symbolic reward.
    Mapping is using ONLY following fields of user_action:
    source.page_id, target.page_id, user_action.action.order.

    Args:
        user_action: user action model to map to a symbolic reward

    Returns:
        symbolic_reward: reward symbol - to be later mapped to a numeric value

    For now it just return generic reward id.
    """
    transition_rewards_df = pd.read_csv(TRANSITION_REWARDS_CSV_PATH, index_col="source")

    source = _to_abstract_page_id(user_action.source.page_id)
    if user_action.action.order:
        target = "order"
    else:
        target = _to_abstract_page_id(user_action.target.page_id)

    symbolic_reward = transition_rewards_df.loc[source, target]
    return symbolic_reward
