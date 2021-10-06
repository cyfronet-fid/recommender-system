# pylint: disable=missing-function-docstring, no-self-use

"""Recommendations endpoint definition"""
import copy

from dotenv import dotenv_values, find_dotenv
from flask import request
from flask_restx import Resource, Namespace

from recommender.engine.agents.pre_agent.pre_agent import PRE_AGENT, PreAgent
from recommender.engine.agents.rl_agent.rl_agent import RL_AGENT, RLAgent
from recommender.errors import InsufficientRecommendationSpace
from recommender.api.schemas.recommendation import (
    recommendation,
    recommendation_context,
)
from recommender.services.deserializer import Deserializer

api = Namespace("recommendations", "Endpoint used for getting recommendations")


def load_agent():
    agent_version = dotenv_values(find_dotenv()).get("AGENT_VERSION")
    agents = {PRE_AGENT: PreAgent, RL_AGENT: RLAgent}
    agent = agents.get(agent_version, PreAgent)()

    return agent


@api.route("")
class Recommendation(Resource):
    """Allows to get recommended services for specific recommendation context"""

    @api.expect(recommendation_context, validate=True)
    @api.response(200, "Recommendations fetched successfully", recommendation)
    def post(self):
        """Returns list of ids of recommended scientific services"""

        json_dict = request.get_json()
        agent = load_agent()
        try:
            services_ids = agent.call(json_dict)

            json_dict_with_services = copy.deepcopy(json_dict)
            json_dict_with_services["services"] = services_ids
            Deserializer.deserialize_recommendation(json_dict_with_services).save()

            response = {"recommendations": services_ids}
        except InsufficientRecommendationSpace:
            response = {"recommendations": []}

        return response
