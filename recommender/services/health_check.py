# pylint: disable=broad-except, too-few-public-methods, protected-access, fixme
"""Module grouping methods for checking the application health"""

from __future__ import annotations

from typing import List, Union, Dict, Optional

import celery
import kombu
import pymongo
import stomp
from flask import current_app
from mongoengine import Document

from recommender.engines.base.base_inference_component import BaseInferenceComponent

from recommender.engines.engines import ENGINES
from recommender.errors import NoSavedMLComponentError
from recommender.models import Service, User
from recommender.tasks import ping

# TODO: adjust if needed
HEALTHY_USERS_THRESHOLD = 10
# TODO: adjust if needed
HEALTHY_SERVICES_THRESHOLD = 10
K = 3


class HealthStatus:
    """
    Class representing a health status of a component.
    Components form a tree-like structure.
    Can be a leaf (specifies status explicitly) or
        a node (status is healthy if all the children are healthy)
    """

    name: str
    status: bool
    error: Optional[str]
    components: List[HealthStatus]

    def __init__(
        self,
        name: str,
        status_or_components: Union[bool, List[HealthStatus]],
        error: Optional[str] = None,
    ):
        self.name = name
        self.error = error

        if isinstance(status_or_components, bool):
            self.status = status_or_components
            self.components = []
        else:
            self.status = all(component.status for component in status_or_components)
            self.components = status_or_components

    def to_dict(self) -> Dict:
        """Report the health status as dict."""
        as_dict = {"status": "UP" if self.status else "DOWN"}
        as_dict |= {child.name: child.to_dict() for child in self.components}
        as_dict |= {"error": self.error} if self.error else {}

        return as_dict


def deep_health_check() -> Dict:
    """
    Perform a deep health check of the recommender application. Checks for:
    - database status (users and services tables have enough members)
    - celery workers (celery worker is running and responds to scheduled tasks
    - databus connection (stomp client has a connection to the databus endpoint)
    - recommender engines (all components can be initialized and loaded)
    """
    return HealthStatus(
        "service",
        [
            HealthStatus(
                "database",
                [
                    _check_tables_health(User, HEALTHY_USERS_THRESHOLD),
                    _check_tables_health(Service, HEALTHY_SERVICES_THRESHOLD),
                ],
            ),
            _check_celery_workers(),
            _check_databus_connection(),
            HealthStatus(
                "recommender_engines",
                [
                    _check_engines_health(engine_name, engine_type)
                    for engine_name, engine_type in ENGINES.items()
                ],
            ),
        ],
    ).to_dict()


def _check_celery_workers() -> HealthStatus:
    name = "celery_worker"
    try:
        ping.apply_async(retry=False, time_limit=3).get(timeout=3)
    except celery.exceptions.TimeoutError:
        return HealthStatus(name, False, "Couldn't reach a celery worker")
    except kombu.exceptions.OperationalError:
        return HealthStatus(name, False, "Redis backend is not operational")

    return HealthStatus(name, True)


def _check_tables_health(
    model: Document.__class__, rows_threshold: int
) -> HealthStatus:
    name = f"{model.__name__}_table".lower()
    try:
        count = model.objects.count()
    except pymongo.errors.ServerSelectionTimeoutError:
        return HealthStatus(name, False, "Error during fetching items")

    if count < rows_threshold:
        return HealthStatus(name, False, "Too few items")

    return HealthStatus(name, True)


def _check_engines_health(
    engine_name: str, engine_type: BaseInferenceComponent.__class__
) -> HealthStatus:
    try:
        engine_type(K=K)
    except NoSavedMLComponentError:
        # If we do not have any engine models in the DB,
        # then the engine is unhealthy
        return HealthStatus(engine_name, False, "Missing model in db")
    return HealthStatus(engine_name, True)


def _check_databus_connection() -> HealthStatus:
    name = "databus"
    try:
        host = current_app.config["RS_DATABUS_HOST"]
        port = current_app.config["RS_DATABUS_PORT"]
        username = current_app.config["RS_DATABUS_USERNAME"]
        password = current_app.config["RS_DATABUS_PASSWORD"]
        enable_ssl = current_app.config["RS_DATABUS_SSL"]
        connection = stomp.Connection([(host, port)])
        connection.connect(
            username=username, password=password, wait=True, ssl=enable_ssl
        )
        connection.disconnect()
    except stomp.exception.ConnectFailedException:
        return HealthStatus(name, False, "Connection to databus failed")
    return HealthStatus(name, True)
