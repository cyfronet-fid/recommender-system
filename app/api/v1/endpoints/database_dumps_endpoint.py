# pylint: disable=missing-function-docstring, no-self-use, unused-variable

"""user action endpoint definition"""

from flask import request
from flask_restx import Resource
from app.api.v1.api import api
from app.api.v1.models.database_dump_model import database_dump
from app.services.mp_dump import load_mp_dump, drop_mp_dump

database_dumps_name_space = api.namespace(
    "database_dumps", "Endpoint used for sending a database dump"
)


@database_dumps_name_space.route("")
class DatabaseDump(Resource):
    """Allows to send database dumps"""

    @api.expect(database_dump, validate=True)
    @api.response(204, "Database dump successfully sent")
    def post(self):
        data = request.get_json()
        # TODO: For now we are dropping the old DB
        drop_mp_dump()
        load_mp_dump(data)
        return None, 204
