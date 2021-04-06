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
from recommender.engine.pre_agent.models.common import (
    load_last_module,
    NoSavedModuleError,
)
from recommender.services.fts import retrieve_services


def _get_not_accessed_services(user):
    ordered_services_ids = [s.id for s in user.accessed_services]
    return Service.objects(id__nin=ordered_services_ids)


def _services_to_ids(services):
    return [s.id for s in services]


def _fill_candidate_services(candidate_services, required_services_no):
    """Fallback in case of len of the candidate_services being too small"""
    if len(candidate_services) < required_services_no:
        diff = required_services_no - len(candidate_services)
        sampled = random.sample(list(Service.objects), diff)
        candidate_services += sampled

    return candidate_services


class PreAgentRecommender:
    """Pre-Agent Recommender based on Neural Collaborative Filtering"""

    def __init__(self, neural_cf_model=None):
        self.neural_cf_model = neural_cf_model

    def call(self, context):
        """This function allows for getting recommended services for the
        recommendation endpoint based on recommendation context
        """

        if self.neural_cf_model is None:
            try:
                self.neural_cf_model = load_last_module(NEURAL_CF)
            except NoSavedModuleError as no_saved_module:
                raise UntrainedPreAgentError from no_saved_module

        k = PANEL_ID_TO_K.get(context["panel_id"])
        if k is None:
            raise InvalidRecommendationPanelIDError

        search_data = context.get("search_data")

        user = None
        if context.get("user_id"):
            user = User.objects(id=context.get("user_id")).first()

        if user:
            return self._for_logged_user(user, search_data, k)
        else:
            return self._for_anonymous_user(search_data, k)

    def _for_logged_user(self, user, search_data, k):
        candidate_services = list(
            set(retrieve_services(search_data)) & set(_get_not_accessed_services(user))
        )

        candidate_services = _fill_candidate_services(candidate_services, k)
        recommended_services_ids = _services_to_ids(candidate_services)

        (
            users_ids,
            users_tensor,
            services_ids,
            services_tensor,
        ) = user_and_services_to_tensors(user, candidate_services)

        matching_probs = self.neural_cf_model(
            users_ids, users_tensor, services_ids, services_tensor
        )
        matching_probs = torch.reshape(matching_probs, (-1,)).tolist()
        top_k = sorted(
            list(zip(matching_probs, recommended_services_ids)), reverse=True
        )[:k]

        return [pair[1] for pair in top_k]

    def _for_anonymous_user(self, search_data, k):
        candidate_services = list(retrieve_services(search_data))
        candidate_services = _fill_candidate_services(candidate_services, k)
        recommended_services = random.sample(list(candidate_services), k)
        return _services_to_ids(recommended_services)


class RecommendationEngineError(Exception):
    pass


class InvalidRecommendationPanelIDError(RecommendationEngineError):
    def message(self):  # pragma: no cover
        return "Invalid recommendation panel id error"


class UntrainedPreAgentError(Exception):
    def message(self):  # pragma: no cover
        return (
            "Pre-Agent can't operate without trained Neural Collaborative"
            " Filtering model - train it before Pre-agent usage via "
            "'/training' endpoint"
        )
