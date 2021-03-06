# pylint: disable=missing-function-docstring, no-self-use, unused-variable

"""Database dumps endpoint definition"""

from flask_restx import Resource, Namespace
from recommender.tasks.neural_networks import execute_pre_agent_training

api = Namespace(
    "training", "Endpoint used for training recommender engine neural networks"
)


@api.route("")
class Training(Resource):
    """Allows to train Pre-Agent"""

    @api.response(200, "Database dump successfully sent")
    def get(self):
        execute_pre_agent_training.delay()
        return None, 200
