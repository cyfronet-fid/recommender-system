# pylint: disable=missing-function-docstring, no-self-use

"""Update endpoint definition"""

from flask import request
from flask_restx import Resource, Namespace

from recommender.api.schemas.database_dump import database_dump
from recommender.tasks.db import handle_db_dump
from recommender.tasks.neural_networks import execute_training

api = Namespace("update", "Endpoint used for updating recommender")


@api.route("")
class Update(Resource):
    """Allows to send database dump, execute training and reload agent"""

    @api.expect(database_dump, validate=True)
    @api.response(204, "Update triggered successfully")
    def post(self):
        data = request.get_json()
        chain = handle_db_dump.s(data) | execute_training.s()
        chain()
        return None, 204
