# pylint: disable=missing-function-docstring, no-self-use

"""Recommendations endpoint definition"""
import copy

from flask import request
from flask_restx import Resource, Namespace

from recommender.engine.agents.pre_agent.pre_agent import PRE_AGENT, PreAgent
from recommender.engine.agents.rl_agent.rl_agent import RL_AGENT, RLAgent
from recommender.engine.agents.base_agent import BaseAgent
from recommender.errors import InsufficientRecommendationSpace
from recommender.api.schemas.recommendation import (
    recommendation,
    recommendation_context,
)
from recommender.services.deserializer import Deserializer

api = Namespace("recommendations", "Endpoint used for getting recommendations")


def load_engine(json_dict: dict) -> BaseAgent:
    """
    Load the engine based on 'engine_version' parameter from a query

    Args:
        json_dict: A body from Marketplace's query

    Returns:
        engine: An instance of pre_agent or rl_agent class
    """
    engine_version = json_dict["engine_version"]
    engines = {PRE_AGENT: PreAgent, RL_AGENT: RLAgent}
    engine = engines.get(engine_version, PreAgent)()

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
            services_ids = agent.call(json_dict)

            json_dict_with_services = copy.deepcopy(json_dict)
            json_dict_with_services["services"] = services_ids
            Deserializer.deserialize_recommendation(json_dict_with_services).save()

            response = {"recommendations": services_ids}
        except InsufficientRecommendationSpace:
            response = {"recommendations": []}

        return response
