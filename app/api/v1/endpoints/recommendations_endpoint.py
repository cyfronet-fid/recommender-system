# pylint: disable=missing-class-docstring, missing-function-docstring, function-redefined, no-self-use

"""recommendations endpoint definition"""

from flask import request
from flask_restx import Resource
from app.api.v1.api import api
from app.api.v1.models.recommendation_model import (
    recommendation,
    recommendation_context,
)
from app.recommender_engine_stub import RecommenderEngineStub
from app.deserializer import Deserializer
from app.recommender_engine_stub import InvalidRecommendationPanelIDError

recommendation_name_space = api.namespace(
    "recommendations", "Endpoint used for getting recommendations"
)


@api.errorhandler(InvalidRecommendationPanelIDError)
def handle_root_exception(error):
    return {"message": error.message()}, 400


@recommendation_name_space.route("")
class Recommendation(Resource):
    @api.expect(recommendation_context, validate=True)
    @api.response(200, "Recommendations fetched successfully", recommendation)
    def post(self):
        """Returns list of ids of recommended scientific services"""
        json_dict = request.get_json()
        services_ids = RecommenderEngineStub.get_recommendations(json_dict)
        json_dict["services"] = services_ids

        Deserializer.deserialize_recommendation(json_dict).save()

        return services_ids
