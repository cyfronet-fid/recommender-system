# pylint: disable=missing-module-docstring

import json
from pathlib import Path
from app.api.v1.api import api, blueprint
from app.api.v1.endpoints import recommendation_name_space
from app.api.v1.endpoints import user_actions_name_space
from app.api.v1.endpoints import database_dumps_endpoint
from global_constants import ABS_PROJECT_ROOT_PATH


def use_api(app):
    """This function register blueprint and restplus endpoints for flask application"""
    app.register_blueprint(blueprint)
    api.add_namespace(recommendation_name_space)
    api.add_namespace(user_actions_name_space)
    api.add_namespace(user_actions_name_space)

    schema_dump_path = (
        Path.joinpath(ABS_PROJECT_ROOT_PATH)
        / "app"
        / "api"
        / "v1"
        / "models"
        / "schemas"
        / "schema_dump.json"
    )
    with open(schema_dump_path, "wt") as file, app.app_context():
        json.dump(api.__schema__, file, indent=4)
