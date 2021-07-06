# pylint: disable-all


def test_training(client, mocker):
    training_task_mock = mocker.patch(
        "recommender.tasks.neural_networks.execute_training.delay"
    )
    response = client.get("/training")
    assert training_task_mock.called
    assert response.status_code == 200
