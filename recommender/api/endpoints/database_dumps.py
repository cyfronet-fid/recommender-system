# pylint: disable=missing-function-docstring, no-self-use, unused-variable

"""Database dumps endpoint definition
"""

from flask import request
from flask_restx import Resource, Namespace

from recommender.api.schemas.database_dump import database_dump
from recommender.services.mp_dump import load_mp_dump, drop_mp_dump

api = Namespace("database_dumps", "Endpoint used for sending a database dump")


@api.route("")
class DatabaseDump(Resource):
    """Allows to send database dumps"""

    @api.expect(database_dump, validate=True)
    @api.response(204, "Database dump successfully sent")
    def post(self):
        data = request.get_json()
        drop_mp_dump()  # For now we are dropping the old DB
        load_mp_dump(data)
        return None, 204
