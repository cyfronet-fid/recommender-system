# pylint: disable=missing-function-docstring

"""User action endpoint definition"""
from typing import Dict

from flask import request
from flask_restx import Resource, Namespace

from recommender.api.schemas.user_action import user_action
from recommender.services.deserializer import Deserializer

api = Namespace("user_actions", "Endpoint used for sending user actions")


def is_service_ua_from_marketplace(json_dict: Dict) -> bool:
    return (
        json_dict.get("client_id") == "marketplace"
        and json_dict.get("source", {}).get("root", {}).get("resource_type")
        == "service"
    )


@api.route("")
class UserAction(Resource):
    """Allows to send user actions"""

    @api.expect(user_action, validate=True)
    @api.response(204, "User action successfully sent")
    def post(self):
        json_dict = request.get_json()
        if is_service_ua_from_marketplace(json_dict):
            Deserializer.deserialize_user_action(json_dict).save()

        return None, 204
