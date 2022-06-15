"""Simple wrapper for entrypoint to jms subscriber"""
import uuid
from typing import Optional


# pylint: disable=too-many-arguments
from jms.connector import subscribe_to_databus


def subscribe(
    host: str,
    port: int,
    username: str,
    password: str,
    topic: str,
    subscription_id: Optional[str] = None,
    ssl: bool = True,
):
    """Simple wrapper function for subscription to databus"""
    if subscription_id is None:
        subscription_id = f"recommender-{uuid.uuid4()}"
    subscribe_to_databus(
        host, port, username, password, topic, subscription_id=subscription_id, ssl=ssl
    )
