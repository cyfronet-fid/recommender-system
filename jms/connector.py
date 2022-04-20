"""Mini framework for JMS interaction"""
import json
import logging
import time
from typing import Optional, Callable

import stomp
from stomp.exception import ConnectFailedException

from logger_config import get_logger
from recommender.tasks import add_user_action

logger = get_logger(__name__)


class UserActionsListener(stomp.ConnectionListener):
    """Main listener handling messages coming from the databus"""

    def on_error(self, frame):
        # Integrate with sentry ?
        logger.error("received an error '%s'", frame.body)

    def on_message(self, frame):
        add_user_action(json.loads(frame.body))


def default_subscription_condition(connection: stomp.Connection) -> bool:
    """Default condition, it can be overwritten in `subscribe_to_databus`"""
    return connection.is_connected()


# pylint: disable=too-many-arguments
def subscribe_to_databus(
    host: str,
    port: int,
    username: str,
    password: str,
    topic: str,
    subscription_id: str,
    ssl: bool = True,
    _logger: Optional[logging.Logger] = None,
    subscription_condition: Optional[Callable] = default_subscription_condition,
) -> None:
    """
    Subscribe to the databus and block until kill signal is issued
    """
    connection = stomp.Connection([(host, port)])
    connection.set_listener("", UserActionsListener())

    if _logger is None:
        _logger = logging.getLogger("cli")
    try:
        connection.connect(username=username, password=password, wait=True, ssl=ssl)
        connection.subscribe(destination=topic, id=subscription_id, ack="auto")
        _logger.info(
            "Subscribed to %(topic)s on %(host)s:%(port)s",
            {"topic": topic, "host": host, "port": port},
        )
        while subscription_condition(connection):
            time.sleep(1.0)
    except ConnectFailedException:
        _logger.critical("Could not subscribe (check host / credentials)")
    except KeyboardInterrupt:
        connection.disconnect()
