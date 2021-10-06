# pylint: disable-all

import json


def test_update_endpoint(client, mp_dump_data, mocker):
    handle_db_dump_s = mocker.patch("recommender.tasks.db.handle_db_dump.s")
    execute_training_s = mocker.patch(
        "recommender.tasks.neural_networks.execute_training.s"
    )

    response = client.post(
        "/update",
        data=json.dumps(mp_dump_data),
        content_type="application/json",
    )

    handle_db_dump_s.assert_called_once_with(mp_dump_data)
    assert execute_training_s.called
    assert response.status_code == 204
