# pylint: disable-all

from recommender.tasks import send_recommendation_to_databus


def test_send_recommendations_to_databus(_app, mocker):
    with _app.app_context():
        mocker.patch("recommender.tasks.stomp.Connection.__init__", return_value=None)
        mocker.patch("recommender.tasks.stomp.Connection.connect")
        mocker.patch("recommender.tasks.stomp.Connection.send")
        mocker.patch("recommender.tasks.stomp.Connection.disconnect")

        context = {"a": 1}
        recommender_response = {"b": 2}

        send_recommendation_to_databus(context, recommender_response)
