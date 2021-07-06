# pylint: disable=missing-function-docstring, no-self-use, unused-variable

"""Reload endpoint definition"""

from flask_restx import Resource, Namespace

from recommender.tasks.neural_networks import reload


api = Namespace(
    "reload", "Reload agent"
)

@api.route("")
class Reload(Resource):
    """Allows to reload agent"""

    def get(self):
        reload.delay()
        return None, 200
