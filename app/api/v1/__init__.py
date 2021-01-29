# pylint: disable=missing-module-docstring

from app.api.v1.api import api, blueprint
from app.api.v1.endpoints import recommendation_name_space
from app.api.v1.endpoints import user_actions_name_space
from app.api.v1.endpoints import database_dumps


def use_api(app):
    """This function register blueprint and restplus endpoints for flask application"""
    app.register_blueprint(blueprint)
    api.add_namespace(recommendation_name_space)
    api.add_namespace(user_actions_name_space)
    api.add_namespace(user_actions_name_space)
