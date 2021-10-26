# pylint: disable-all

import pandas as pd
import pytest

from recommender.engines.rl.ml_components.reward_mapping import (
    TRANSITION_REWARDS_CSV_PATH,
)
from recommender.engines.rl.ml_components.synthetic_dataset.rewards import (
    _get_closest_reward,
    _draw_reward,
    synthesize_reward,
    RewardGeneration,
)


@pytest.fixture()
def transitions_df():
    return pd.read_csv(TRANSITION_REWARDS_CSV_PATH, index_col="source")


def test__get_closest_reward():
    assert (
        _get_closest_reward("order", ["exit", "simple_transition", "mild_interest"])
        == "mild_interest"
    )
    assert (
        _get_closest_reward("order", ["mild_interest", "exit", "simple_transition"])
        == "mild_interest"
    )
    assert _get_closest_reward("order", []) == "exit"
    assert (
        _get_closest_reward(
            "order", ["mild_interest", "order", "exit", "simple_transition", "interest"]
        )
        == "order"
    )
    assert (
        _get_closest_reward(
            "mild_interest",
            ["mild_interest", "order", "exit", "simple_transition", "interest"],
        )
        == "mild_interest"
    )
    assert (
        _get_closest_reward("exit", ["mild_interest", "exit", "simple_transition"])
        == "exit"
    )


def test__draw_reward():
    repetitions = 1000

    high_engagement_buffer = []
    low_engagement_buffer = []

    for _ in range(repetitions):
        high_engagement_buffer.append(_draw_reward(1.0))
        low_engagement_buffer.append(_draw_reward(0.0))

    he_ordered_ratio = (
        len(list(filter(lambda x: x == "order", high_engagement_buffer))) / repetitions
    )
    he_exit_ratio = (
        len(list(filter(lambda x: x == "exit", high_engagement_buffer))) / repetitions
    )
    le_ordered_ratio = (
        len(list(filter(lambda x: x == "order", low_engagement_buffer))) / repetitions
    )
    le_exit_ratio = (
        len(list(filter(lambda x: x == "exit", low_engagement_buffer))) / repetitions
    )

    assert he_ordered_ratio > 0.8
    assert he_exit_ratio < 0.2
    assert le_ordered_ratio < 0.2
    assert le_exit_ratio > 0.8


def test_construct_rewards(transitions_df):
    repetitions = 1000

    high_engagement_buffer = []

    for _ in range(repetitions):
        high_engagement_buffer.append(synthesize_reward(transitions_df, 1.0))

    he_orders_percent = (
        len(
            list(
                filter(
                    lambda x: len(x) > 0 and x[-1] == "order", high_engagement_buffer
                )
            )
        )
        / repetitions
    )
    he_empty_percent = (
        len(list(filter(lambda x: len(x) == 0, high_engagement_buffer))) / repetitions
    )

    assert he_orders_percent > 0.8
    assert he_empty_percent < 0.2

    low_engagement_buffer = []

    for _ in range(repetitions):
        low_engagement_buffer.append(synthesize_reward(transitions_df, 0.0))

    le_orders_percent = (
        len(
            list(
                filter(lambda x: len(x) > 0 and x[-1] == "order", low_engagement_buffer)
            )
        )
        / repetitions
    )
    le_empty_percent = (
        len(list(filter(lambda x: len(x) == 0, low_engagement_buffer))) / repetitions
    )

    assert le_orders_percent < 0.2
    assert le_empty_percent > 0.8


def test_proper_simple_initialization(transitions_df):
    with pytest.raises(AssertionError):
        synthesize_reward(
            transitions_df, 0.0, mode=RewardGeneration.SIMPLE, simple_mode_threshold=1.2
        )
        synthesize_reward(
            transitions_df,
            0.0,
            mode=RewardGeneration.SIMPLE,
            simple_mode_threshold=-0.1,
        )


def test_synthesize_simple_reward(transitions_df):
    assert (
        synthesize_reward(
            transitions_df, 0.2, mode=RewardGeneration.SIMPLE, simple_mode_threshold=0.5
        )
        == []
    )
    assert synthesize_reward(
        transitions_df, 0.7, mode=RewardGeneration.SIMPLE, simple_mode_threshold=0.5
    ) == ["order"]
