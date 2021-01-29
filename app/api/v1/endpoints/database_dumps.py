# pylint: disable=missing-function-docstring, no-self-use, unused-variable

"""user action endpoint definition"""

from flask import request
from flask_restplus import Resource
from app.api.v1.api import api
from app.api.v1.models.database_dumps import database_dump

database_dumps_name_space = api.namespace(
    "database_dumps", "Endpoint used for sending a database dump"
)


@database_dumps_name_space.route("")
class DatabaseDump(Resource):
    """Allows to send database dumps"""

    @api.expect(database_dump, validate=True)
    @api.response(204, "Database dump successfully sent")
    def post(self):
        databasedump = request.get_json()
        # save dump code

        return None, 204
