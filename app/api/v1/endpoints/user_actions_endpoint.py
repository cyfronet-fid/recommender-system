# pylint: disable=missing-function-docstring, no-self-use

"""user action endpoint definition"""

from flask import request
from flask_restx import Resource
from app.api.v1.api import api
from app.api.v1.models.user_action_model import user_action
from app.user_activity_gatherer import UserActivityGatherer

user_actions_name_space = api.namespace(
    "user_actions", "Endpoint used for sending user actions"
)


@user_actions_name_space.route("")
class UserAction(Resource):
    """Allows to send user actions"""

    @api.expect(user_action, validate=True)
    @api.response(204, "User action successfully sent")
    def post(self):
        useraction = request.get_json()
        UserActivityGatherer.save_user_action(useraction)

        return None, 204
