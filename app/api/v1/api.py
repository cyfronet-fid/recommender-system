# pylint: disable=missing-module-docstring

from flask import Blueprint
from flask_restx import Api


blueprint = Blueprint("api_v1", __name__, url_prefix="/api/v1")
api = Api(
    blueprint,
    doc="/doc/",
    version="1.0",
    title="Recommender system API",
    description="Recommender System API for getting recommendations, sending user \
    actions, sending Marketplace database dumps and triggering recommender system \
    offline learning.",
)
