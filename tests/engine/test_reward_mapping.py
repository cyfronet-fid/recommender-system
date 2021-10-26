# pylint: disable-all
import pytest

from recommender.engines.rl.ml_components.reward_mapping import (
    _to_abstract_page_id,
    ua_to_reward_id,
)
from tests.factories.user_action import UserActionFactory
from tests.factories.user_action.action import ActionFactory
from tests.factories.user_action.source import SourceFactory
from tests.factories.user_action.target import TargetFactory


@pytest.fixture
def valid_page_ids():
    return [
        "/services",
        "/services/{id}",
        "/services/{id}/details",
        "/services/{id}/opinions",
        "/services/{id}/offers",
        "/services/{id}/information",
        "/services/{id}/configuration",
        "/services/{id}/summary",
        "/comparisons",
    ]


def test__to_abstract_page_id(valid_page_ids):
    assert (
        _to_abstract_page_id("/services/egi-cloud-compute", valid_page_ids)
        == "/services/{id}"
    )
    assert (
        _to_abstract_page_id("/services/b2access/information", valid_page_ids)
        == "/services/{id}/information"
    )
    assert (
        _to_abstract_page_id("/services/b2access/summary", valid_page_ids)
        == "/services/{id}/summary"
    )
    assert _to_abstract_page_id("/services", valid_page_ids) == "/services"
    assert _to_abstract_page_id("/services/c/compute", valid_page_ids) == "/services"
    assert (
        _to_abstract_page_id("/services/c/other-other", valid_page_ids) == "/services"
    )
    assert _to_abstract_page_id("/comparisons", valid_page_ids) == "/comparisons"

    assert _to_abstract_page_id("/comparisons/asd", valid_page_ids) == "unknown_page_id"
    assert _to_abstract_page_id("/gibberish", valid_page_ids) == "unknown_page_id"
    assert _to_abstract_page_id("XD", valid_page_ids) == "unknown_page_id"
    assert (
        _to_abstract_page_id("/services/b2access/some-page", valid_page_ids)
        == "unknown_page_id"
    )
    assert (
        _to_abstract_page_id("/projects/1449/asd", valid_page_ids) == "unknown_page_id"
    )
    assert _to_abstract_page_id("/", valid_page_ids) == "unknown_page_id"
    assert _to_abstract_page_id("", valid_page_ids) == "unknown_page_id"


def test_ua_to_reward_id(mongo):
    assert (
        ua_to_reward_id(
            UserActionFactory(
                source=SourceFactory(page_id="/services/b2access"),
                target=TargetFactory(page_id="/services/b2access"),
            )
        )
        == "simple_transition"
    )

    assert (
        ua_to_reward_id(
            UserActionFactory(
                source=SourceFactory(page_id="/services/b2access"),
                target=TargetFactory(page_id="/services/b2access/information"),
            )
        )
        == "interest"
    )

    assert (
        ua_to_reward_id(
            UserActionFactory(
                source=SourceFactory(page_id="/services/b2access"),
                target=TargetFactory(page_id="/services/b2access/summary"),
            )
        )
        == "unknown_transition"
    )

    assert (
        ua_to_reward_id(
            UserActionFactory(
                source=SourceFactory(page_id="/services/b2access/summary"),
                action=ActionFactory(order=True),
            )
        )
        == "order"
    )

    assert (
        ua_to_reward_id(
            UserActionFactory(
                source=SourceFactory(page_id="/comparison"),
                target=TargetFactory(page_id="/not-yet-implemented"),
            )
        )
        == "unknown_transition"
    )

    assert (
        ua_to_reward_id(
            UserActionFactory(
                source=SourceFactory(page_id="/not-yet-implemented"),
                target=TargetFactory(page_id="/services/b2access/information"),
            )
        )
        == "interest"
    )

    assert (
        ua_to_reward_id(
            UserActionFactory(
                source=SourceFactory(page_id="/not-yet-implemented"),
                action=ActionFactory(order=True),
            )
        )
        == "order"
    )

    assert (
        ua_to_reward_id(
            UserActionFactory(
                source=SourceFactory(page_id="/services/c/compute-other"),
                target=TargetFactory(page_id="/services/egi-cloud-compute"),
            )
        )
        == "mild_interest"
    )
