# pylint: disable=invalid-name, no-member, missing-module-docstring

from typing import List

import numpy as np
import pandas as pd

from enum import Enum, auto


class RewardGeneration(Enum):
    SIMPLE = auto()
    COMPLEX = auto()


# NOTE: that rewards are listed in increasing interest order
REWARDS = ["exit", "simple_transition", "mild_interest", "interest", "order"]


def _draw_reward(engagement):
    """Draws reward based on engagement. It is based on binomial distribution.
    The intuition is that the higher the engagement,
    the easier it is for the distribution to pick
    high-engagement action like interest or order and vice-versa."""
    reward = np.random.binomial(len(REWARDS) - 1, engagement)
    return REWARDS[reward]


def _get_closest_reward(required_reward, available_rewards):
    """Returns reward closest to the required reward in terms of engagement"""
    for r in reversed(REWARDS[: REWARDS.index(required_reward) + 1]):
        if r in available_rewards:
            return r
    return "exit"


def synthesize_reward(
    transitions_df: pd.DataFrame,
    engagement: float,
    max_depth: int = 10,
    current_depth: int = 0,
    source: str = "/services",
    mode: RewardGeneration = RewardGeneration.COMPLEX,
    simple_mode_threshold: float = 0.5,
) -> List[str]:
    """
    Synthesizes reward using service_engagement factor.
    If mode is "simple" - the synthesized reward is either ["order"] or [] depending on the
    specified threshold. In the "complex" mode case,
    it recursively explores the transition graph,
    and chooses appropriate rewards based on engagement
    and binomial distribution of engagement over rewards.

    Args:
        transitions_df: transition adjacency matrix, with reward mapping
        engagement: service engagement that is returned by
            approx_service_engagement (between 0 and 1)
        max_depth: max depth of the recursive stack
        current_depth: current depth of the stack, should always be 0 on the first call
        source: starting page, should always equal to /services on the first call
        mode: either "complex" or "simple"
        simple_mode_threshold - specifies above which value the synthesized reward should be "order"

    Returns:
        reward list: list of symbolic rewards given for a given graph walk
    """

    assert 0.0 <= simple_mode_threshold <= 1.0

    if mode == RewardGeneration.SIMPLE:
        return ["order"] if engagement > simple_mode_threshold else []

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

    return [drawn_reward] + synthesize_reward(
        transitions_df, engagement, max_depth, current_depth + 1, source=drawn_target
    )
