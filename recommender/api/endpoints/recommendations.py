# pylint: disable=missing-function-docstring, no-self-use, line-too-long, fixme
# pylint: disable=invalid-name

"""Recommendations endpoint definition"""
import copy
from typing import Dict, Any

from flask import request
from flask_restx import Resource, Namespace

from recommender.engines.base.base_inference_component import BaseInferenceComponent
from recommender.engines.ncf.inference.ncf_inference_component import (
    NCFInferenceComponent,
)
from recommender.engines.panel_id_to_services_number_mapping import PANEL_ID_TO_K
from recommender.engines.rl.inference.rl_inference_component import RLInferenceComponent
from recommender.errors import (
    InsufficientRecommendationSpace,
    InvalidRecommendationPanelIDError,
)
from recommender.api.schemas.recommendation import (
    recommendation,
    recommendation_context,
)
from recommender.services.deserializer import Deserializer

api = Namespace("recommendations", "Endpoint used for getting recommendations")


def get_K(context: Dict[str, Any]) -> int:
    """
    Get the K constant from the context.

    Args:
        context: context json  from the /recommendations endpoint request.

    Returns:
        K constant.
    """
    K = PANEL_ID_TO_K.get(context.get("panel_id"))
    if K is None:
        raise InvalidRecommendationPanelIDError
    return K


def load_engine(json_dict: dict) -> BaseInferenceComponent:
    """
    Load the engine based on 'engine_version' parameter from a query

    Args:
        json_dict: A body from Marketplace's query

    Returns:
        engine: An instance of NCFInferenceComponent or RLInferenceComponent
    """
    engine_version = json_dict.get("engine_version")
    K = get_K(json_dict)

    engines = {
        NCFInferenceComponent.__name__: NCFInferenceComponent,
        RLInferenceComponent.__name__: RLInferenceComponent,
    }
    engine = engines.get(engine_version, RLInferenceComponent)(K)
    return engine


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
        except InsufficientRecommendationSpace:
            response = {"recommendations": []}

        return response
