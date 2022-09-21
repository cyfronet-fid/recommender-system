# pylint: disable-all

import json
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
    user_action_data["client_id"] = "marketplace"

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
