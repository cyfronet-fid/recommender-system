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
from recommender.services.provide_service_ctx import service_ctx
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

        json_dict = service_ctx(request.get_json())
        engine, engine_name = load_engine(json_dict)
        panel_id = json_dict.get("panel_id")
        try:
            services_ids, scores, explanations = engine(json_dict)
            explanations_long, explanations_short = [
                list(t) for t in list(zip(*[(e.long, e.short) for e in explanations]))
            ]
            deserialize_recommendation(json_dict, services_ids, engine_name)

            response = {
                "panel_name": panel_id,
                "recommendations": services_ids,
                "explanations": explanations_long,
                "explanations_short": explanations_short,
                "score": scores,
                # Beter name is "scores" as it's a list - not a single
                # value, so internally we will use scores. We can consider
                # to change the external API from "score" to "scores" too?
            }

        except InsufficientRecommendationSpaceError:
            logger.error(InsufficientRecommendationSpaceError().message())
            response = {"recommendations": []}

        return response
