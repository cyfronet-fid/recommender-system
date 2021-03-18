# pylint: disable-all

import json

from pytest import fixture


@fixture
def user_action_data():
    return {
        "logged_user": True,
        "user_id": 1234,
        "unique_id": 1234,
        "timestamp": "2021-03-18T19:33:21.620Z",
        "source": {
            "visit_id": 1234,
            "page_id": "services_catalogue_list",
            "root": {
                "type": "recommendation_panel",
                "panel_id": "v1",
                "service_id": 1234,
            },
        },
        "target": {"visit_id": 1234, "page_id": "service_about"},
        "action": {"type": "button", "text": "Details", "order": True},
    }


def test_user_actions(client, mocker, user_action_data):
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
