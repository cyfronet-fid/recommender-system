# pylint: disable-all

import json
import threading
import time
import uuid
from typing import Optional

import pytest
import stomp
from pytest_mock import MockerFixture
from flask import current_app

from jms.connector import subscribe_to_databus
from recommender.types import UserAction
from tests.factories.requests.user_actions import UserActionFactory

Second = int


class MockSubscriptionCondition:
    def __init__(self) -> None:
        self.terminated = False

    def __call__(self, connection: stomp.Connection) -> bool:
        return not self.terminated

    def terminate(self) -> None:
        self.terminated = True


class JMSSubscriberController:
    THREAD_LATENCY: Second = 2.5

    def __init__(self, host, port, username, password, topic, subscriber_id):
        self.condition = MockSubscriptionCondition()
        self.thread = threading.Thread(
            target=subscribe_to_databus,
            args=(host, port, username, password, topic, subscriber_id),
            kwargs=dict(subscription_condition=self.condition),
        )
        self.topic = topic
        self.client = stomp.Connection(host_and_ports=[(host, port)])
        self.client.connect(username, password, wait=True)

    def send(self, data: dict, topic: Optional[str] = None) -> None:
        self.client.send(
            self.topic if topic is None else topic,
            json.dumps(data),
            content_type="application/json",
        )
        # wait for listeners to accept the message
        # this makes testing easier, as developer does not have to remember to wait after
        # every send
        time.sleep(self.THREAD_LATENCY)

    def terminate(self) -> None:
        self.condition.terminate()
        self.thread.join()

    def start(self) -> None:
        self.thread.start()
        # wait for listeners to start
        time.sleep(self.THREAD_LATENCY)


@pytest.fixture
def jms_controller(_app) -> JMSSubscriberController:
    with _app.app_context():
        controller = JMSSubscriberController(
            host=current_app.config["RS_DATABUS_HOST"],
            port=current_app.config["RS_DATABUS_PORT"],
            username=current_app.config["RS_DATABUS_USERNAME"],
            password=current_app.config["RS_DATABUS_PASSWORD"],
            topic=current_app.config["RS_DATABUS_SUBSCRIPTION_TOPIC"],
            subscriber_id=str(uuid.uuid4()),
        )

    controller.start()
    yield controller
    controller.terminate()


def test_listener(jms_controller, mocker: MockerFixture):
    deserializer_mock = mocker.patch(
        "recommender.tasks.Deserializer.deserialize_user_action"
    )

    user_action_data = UserActionFactory()
    jms_controller.send(user_action_data)
    deserializer_mock.assert_called_once_with(UserAction.parse_obj(user_action_data))
