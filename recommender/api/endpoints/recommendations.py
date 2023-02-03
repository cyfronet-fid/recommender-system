"""Recommendations endpoint definition"""

from flask import request
from flask_restx import Resource, Namespace

from recommender.errors import (
    InsufficientRecommendationSpaceError,
)
from recommender.api.schemas.recommendation import (
    recommendation,
    recommendation_context,
)
from recommender.services.provide_ctx import provide_ctx
from recommender.services.deserializer import deserialize_recommendation
from recommender.services.engine_selector import load_engine
from recommender.tasks import send_recommendation_to_databus
from logger_config import get_logger

api = Namespace("recommendations", "Endpoint used for getting recommendations")
logger = get_logger(__name__)


@api.route("")
class Recommendation(Resource):
    """Allows to get recommended services for specific recommendation context"""

    @api.expect(recommendation_context, validate=True)
    @api.response(200, "Recommendations fetched successfully", recommendation)
    def post(self):
        """Returns list of ids of recommended scientific services"""
        req_body = provide_ctx(request.get_json())
        try:
            response = self.process_request(req_body)

            if req_body.get("client_id") == "marketplace":
                send_recommendation_to_databus.delay(req_body, response)

        except InsufficientRecommendationSpaceError:
            logger.error(InsufficientRecommendationSpaceError().message())
            response = {"recommendations": []}

        return response

    @staticmethod
    def process_request(req_body: dict) -> dict:
        """Process the request and get recommendations"""
        engine, engine_name = load_engine(req_body)
        services_ids, scores, explanations = engine(req_body)

        explanations_long, explanations_short = [
            list(t) for t in list(zip(*[(e.long, e.short) for e in explanations]))
        ]
        deserialize_recommendation(req_body, services_ids, engine_name)

        response = {
            "panel_id": req_body.get("panel_id"),
            "recommendations": services_ids,
            "explanations": explanations_long,
            "explanations_short": explanations_short,
            "scores": scores,
            "engine_version": engine_name,
        }

        return response
