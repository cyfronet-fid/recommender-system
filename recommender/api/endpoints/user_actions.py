# pylint: disable=missing-function-docstring, no-self-use

"""User action endpoint definition"""

from flask import request
from flask_restx import Resource, Namespace

from recommender.api.schemas.user_action import user_action
from recommender.services.deserializer import Deserializer

api = Namespace("user_actions", "Endpoint used for sending user actions")


@api.route("")
class UserAction(Resource):
    """Allows to send user actions"""

    @api.expect(user_action, validate=True)
    @api.response(204, "User action successfully sent")
    def post(self):
        json_dict = request.get_json()
        Deserializer.deserialize_user_action(json_dict).save()

        return None, 204
