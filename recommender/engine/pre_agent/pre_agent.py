# pylint: disable=unused-argument, too-few-public-methods, missing-class-docstring, missing-function-docstring
# pylint: disable=no-self-use, superfluous-parens, no-else-return, fixme, no-member

"""Implementation of the recommender engine Pre-Agent"""

import random
import torch

from recommender.engine.pre_agent.models.neural_colaborative_filtering import NEURAL_CF
from recommender.engine.pre_agent.preprocessing.preprocessing import (
    user_and_services_to_tensors,
)
from recommender.models import User, Service

from recommender.engine.panel_id_to_services_number_mapping import PANEL_ID_TO_K
from recommender.engine.pre_agent.models.common import load_last_module


def _get_not_accessed_services(user):
    ordered_services_ids = [s.id for s in user.accessed_services]
    return Service.objects(id__nin=ordered_services_ids)


def _services_to_ids(services):
    return [s.id for s in services]


class PreAgentRecommender:
    """Pre-Agent Recommender based on Neural Colaborative Filtering"""

    def __init__(self, neural_cf_model=None):
        self.neural_cf_model = neural_cf_model

    def call(self, context):
        """This function allows for getting recommended services for the
        recommendation endpoint based on recommendation context
        """

        if self.neural_cf_model is None:
            self.neural_cf_model = load_last_module(NEURAL_CF)

        k = PANEL_ID_TO_K.get(context["panel_id"])
        if k is None:
            raise InvalidRecommendationPanelIDError
        search_phrase = context.get("search_phrase")
        filters = context.get("search_data")

        user = None
        if context.get("logged_user"):
            user = User.objects(id=context.get("user_id")).first()

        if user:
            return self._for_logged_user(user, search_phrase, filters, k)
        else:
            return self._for_anonymous_user(search_phrase, filters, k)

    def _for_logged_user(self, user, _search_phrase, _filters, k):
        # TODO: use _search_phrase and _filters after elasticsearch integration

        not_accessed_services = _get_not_accessed_services(user)
        services_ids = _services_to_ids(not_accessed_services)

        users_tensor, services_tensor = user_and_services_to_tensors(
            user, not_accessed_services
        )

        matching_probs = self.neural_cf_model(users_tensor, services_tensor)
        matching_probs = torch.reshape(matching_probs, (-1,)).tolist()
        top_k = sorted(list(zip(matching_probs, services_ids)), reverse=True)[:k]

        return [pair[1] for pair in top_k]

    def _for_anonymous_user(self, _search_phrase, _filters, k):
        # TODO: use _search_phrase and _filters after elasticsearch integration

        recommended_services = random.sample(list(Service.objects), k)
        return _services_to_ids(recommended_services)


class RecommendationEngineError(Exception):
    pass


class InvalidRecommendationPanelIDError(RecommendationEngineError):
    def message(self):
        return "Invalid recommendation panel id error"
