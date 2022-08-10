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
from recommender.services.deserializer import deserialize_recommendation
from recommender.services.engine_selector import load_engine
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

        json_dict = request.get_json()
        engine, engine_name = load_engine(json_dict)
        try:
            services_ids = engine(json_dict)
            deserialize_recommendation(json_dict, services_ids, engine_name)

            response = {"recommendations": services_ids}

        except InsufficientRecommendationSpaceError:
            logger.error(InsufficientRecommendationSpaceError().message())
            response = {"recommendations": []}

        return response
