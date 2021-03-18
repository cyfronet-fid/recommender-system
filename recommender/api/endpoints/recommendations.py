# pylint: disable=missing-function-docstring, no-self-use

"""Recommendations endpoint definition"""

from flask import request, current_app
from flask_restx import Resource, Namespace

from recommender.api.schemas.recommendation import (
    recommendation,
    recommendation_context,
)
from recommender.services.deserializer import Deserializer
from recommender.engine.recommender_engine_stub import (
    InvalidRecommendationPanelIDError,
)

api = Namespace("recommendations", "Endpoint used for getting recommendations")


@api.errorhandler(InvalidRecommendationPanelIDError)
def handle_root_exception(error):
    return {"message": error.message()}, 400


@api.route("")
class Recommendation(Resource):
    """Allows to get recommended services for specific recommendation context"""

    @api.expect(recommendation_context, validate=True)
    @api.response(200, "Recommendations fetched successfully", recommendation)
    def post(self):
        """Returns list of ids of recommended scientific services"""

        json_dict = request.get_json()
        with current_app.app_context():
            services_ids = current_app.recommender_engine.call(json_dict)

        json_dict["services"] = services_ids
        Deserializer.deserialize_recommendation(json_dict).save()

        response = {
            "recommendations": services_ids
        }

        return response
