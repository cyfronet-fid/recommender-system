"""Api object definition and endpoint namespaces registration"""

from flask_restx import Api

from recommender.api.schemas import api as models_ns
from recommender.api.endpoints.recommendations import api as recommendations_ns
from recommender.api.endpoints.user_actions import api as user_actions_ns
from recommender.api.endpoints.update import api as update_ns
from recommender.api.endpoints.health_check import api as health_check_ns


api = Api(
    doc="/",
    version="1.0",
    title="Recommender system API",
    description="Recommender System API for getting recommendations, sending user "
    "actions, sending Marketplace database dumps and triggering "
    "recommender system offline learning.",
)

# API namespaces
api.add_namespace(models_ns)
api.add_namespace(recommendations_ns)
api.add_namespace(user_actions_ns)
api.add_namespace(update_ns)
api.add_namespace(health_check_ns)
