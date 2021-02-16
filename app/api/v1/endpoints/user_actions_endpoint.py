# pylint: disable=missing-function-docstring, no-self-use

"""user action endpoint definition"""

from flask import request
from flask_restx import Resource
from app.api.v1.api import api
from app.api.v1.models.user_action_model import user_action
from app.deserializer import Deserializer

user_actions_name_space = api.namespace(
    "user_actions", "Endpoint used for sending user actions"
)


@user_actions_name_space.route("")
class UserAction(Resource):
    """Allows to send user actions"""

    @api.expect(user_action, validate=True)
    @api.response(204, "User action successfully sent")
    def post(self):
        json_dict = request.get_json()
        Deserializer.deserialize_user_action(json_dict).save()

        return None, 204
