"""Health check endpoint definition"""

from __future__ import annotations
from flask_restx import Resource, Namespace
from recommender.services.health_check import deep_health_check

api = Namespace("health", "Endpoint used for checking the application's health")


@api.route("")
class HealthCheck(Resource):
    """Groups methods for checking the application health"""

    def get(self):
        """Perform a deep health_check"""
        return deep_health_check()
