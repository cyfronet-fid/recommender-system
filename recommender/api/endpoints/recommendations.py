# pylint: disable=no-self-use

"""Recommendations endpoint definition"""
import copy

from flask import request
from flask_restx import Resource, Namespace

from recommender.errors import (
    InsufficientRecommendationSpaceError,
)
from recommender.api.schemas.recommendation import (
    recommendation,
    recommendation_context,
)
from recommender.services.deserializer import Deserializer
from recommender.services.rec_engine_selector import load_engine
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
        agent = load_engine(json_dict)
        try:
            services_ids = agent(json_dict)

            json_dict_with_services = copy.deepcopy(json_dict)
            json_dict_with_services["services"] = services_ids
            Deserializer.deserialize_recommendation(json_dict_with_services).save()

            response = {"recommendations": services_ids}
        except InsufficientRecommendationSpaceError:
            logger.error(InsufficientRecommendationSpaceError().message())
            response = {"recommendations": []}

        return response
