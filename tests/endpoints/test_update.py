# pylint: disable-all

import json


def test_update_endpoint(client, mp_dump_data, mocker):
    execute_training_delay = mocker.patch("recommender.tasks.update.delay")

    response = client.post(
        "/update", data=json.dumps(mp_dump_data), content_type="application/json"
    )

    execute_training_delay.assert_called_once_with(mp_dump_data)
    assert execute_training_delay.called
    assert response.status_code == 204
