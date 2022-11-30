# pylint: disable-all

import json

import pytest

from tests.factories.requests.user_actions import UserActionFactory


def test_user_actions(client, mocker):
    user_action_data = UserActionFactory()
    deserializer_mock = mocker.patch(
        "recommender.services.deserializer.Deserializer.deserialize_user_action"
    )
    response = client.post(
        "/user_actions",
        data=json.dumps(user_action_data),
        content_type="application/json",
    )

    deserializer_mock.assert_called_once_with(user_action_data)
    assert response.status_code == 204


def test_user_actions_with_aai_uid(client, mocker):
    user_action = UserActionFactory()
    user_action_data = json.loads(json.dumps(user_action))
    del user_action_data["user_id"]
    user_action_data["aai_uid"] = "abc@egi.eu"

    deserializer_mock = mocker.patch(
        "recommender.services.deserializer.Deserializer.deserialize_user_action"
    )
    response = client.post(
        "/user_actions",
        data=json.dumps(user_action_data),
        content_type="application/json",
    )

    deserializer_mock.assert_called_once_with(user_action_data)
    assert response.status_code == 204


@pytest.mark.parametrize("client_id", ["search_service", "user_dashboard", "undefined"])
def test_user_action_not_from_marketplace_is_not_saved(client_id, client, mocker):
    user_action = UserActionFactory()
    user_action_data = json.loads(json.dumps(user_action))
    user_action_data["client_id"] = client_id
    deserializer_mock = mocker.patch(
        "recommender.services.deserializer.Deserializer.deserialize_user_action"
    )
    response = client.post(
        "/user_actions",
        data=json.dumps(user_action_data),
        content_type="application/json",
    )

    assert not deserializer_mock.called
    assert response.status_code == 204


def test_user_action_with_resource_type_other_than_service_is_not_saved(client, mocker):
    user_action = UserActionFactory()
    user_action_data = json.loads(json.dumps(user_action))
    user_action_data["source"]["root"]["resource_type"] = "placeholder_resource_type"
    deserializer_mock = mocker.patch(
        "recommender.services.deserializer.Deserializer.deserialize_user_action"
    )
    response = client.post(
        "/user_actions",
        data=json.dumps(user_action_data),
        content_type="application/json",
    )

    assert not deserializer_mock.called
    assert response.status_code == 204
