# pylint: disable-all


def test_training(client, mocker):
    pre_training_task_mock = mocker.patch(
        "recommender.tasks.neural_networks.execute_pre_agent_training.delay"
    )
    rl_training_task_mock = mocker.patch(
        "recommender.tasks.neural_networks.execute_rl_agent_training.delay"
    )
    response = client.get("/training")
    assert pre_training_task_mock.called or rl_training_task_mock.called
    assert response.status_code == 200
