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
