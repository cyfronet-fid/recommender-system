# pylint: disable=missing-class-docstring, missing-function-docstring, function-redefined, no-self-use

"""recommendations endpoint definition"""

from flask import request
from flask_restx import Resource
from app.api.v1.api import api
from app.api.v1.models.recommendation_model import (
    recommendation,
    recommendation_context,
    recommendation_parser,
)
from app.recommender_engine_stub import RecommenderEngineStub
from app.user_activity_gatherer import UserActivityGatherer
from app.recommender_engine_stub import (
    InvalidRecommendationPanelLocationError,
    InvalidRecommendationPanelVersionError,
)

recommendation_name_space = api.namespace(
    "recommendations", "Endpoint used for getting recommendations"
)


@api.errorhandler(InvalidRecommendationPanelVersionError)
def handle_root_exception(error):
    return {"message": error.message()}, 400


@api.errorhandler(InvalidRecommendationPanelLocationError)
def handle_root_exception(error):
    return {"message": error.message()}, 400


@recommendation_name_space.route("")
class Recommendation(Resource):
    @api.expect(recommendation_context, recommendation_parser, validate=True)
    @api.response(200, "Successfully fetched recommendations", recommendation)
    def post(self):
        """Returns list of ids of recommended scientific services"""
        args = recommendation_parser.parse_args()
        location = args.get("location", None)
        version = args.get("version", None)
        context = request.get_json()

        full_context = {
            "user_id": context["user_id"],
            "timestamp": "time",
            "filters": context.get("filters"),
            "search_phrase": context.get("search_phrase"),
        }

        recommended_services_ids = RecommenderEngineStub.get_recommendations(
            context, location, version
        )
        UserActivityGatherer.save_state_and_action(
            state=full_context, action=recommended_services_ids
        )

        return (
            {"status": "success", "message": "Recommendations fetch sucessfull"},
            200,
            {},
        )
