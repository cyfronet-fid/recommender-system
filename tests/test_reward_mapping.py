# pylint: disable-all

from recommender.engine.reward_mapping import _to_abstract_page_id, ua_to_reward_id
from tests.factories.user_action import UserActionFactory
from tests.factories.user_action.action import ActionFactory
from tests.factories.user_action.source import SourceFactory
from tests.factories.user_action.target import TargetFactory


def test__to_abstract_page_id():
    assert _to_abstract_page_id("/services/egi-cloud-compute") == "/services/{id}"
    assert (
        _to_abstract_page_id("/services/b2access/information")
        == "/services/{id}/information"
    )
    assert (
        _to_abstract_page_id("/services/b2access/summary") == "/services/{id}/summary"
    )
    assert _to_abstract_page_id("/comparisons") == "/comparisons"
    assert _to_abstract_page_id("/gibberish") == "unknown_page_id"
    assert _to_abstract_page_id("XD") == "unknown_page_id"
    assert _to_abstract_page_id("/services") == "unknown_page_id"
    assert _to_abstract_page_id("/projects/1449/asd") == "unknown_page_id"
    assert _to_abstract_page_id("/") == "unknown_page_id"
    assert _to_abstract_page_id("") == "unknown_page_id"


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
