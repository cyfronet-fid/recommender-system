# pylint: disable-all


def test_training(client, mocker):
    pre_training_task_mock = mocker.patch(
        "recommender.tasks.neural_networks.execute_pre_agent_training.delay"
    )
    response = client.get("/training")
    pre_training_task_mock.assert_called_once()
    assert response.status_code == 200
