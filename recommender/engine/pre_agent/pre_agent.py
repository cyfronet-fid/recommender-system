# pylint: disable=unused-argument, too-few-public-methods, missing-class-docstring, missing-function-docstring
# pylint: disable=no-self-use, superfluous-parens, no-else-return, fixme, no-member

"""Implementation of the recommender engine Pre-Agent"""

from typing import List

import torch

from recommender.models import SearchData
from recommender.engine.base_agent_recommender import BaseAgentRecommender
from recommender.errors import UntrainedPreAgentError
from recommender.engine.pre_agent.models.neural_colaborative_filtering import NEURAL_CF
from recommender.engine.pre_agent.preprocessing.preprocessing import (
    user_and_services_to_tensors,
)
from recommender.models import User

from recommender.engine.pre_agent.models.common import (
    load_last_module,
    NoSavedModuleError,
)
from recommender.services.fts import retrieve_services


class PreAgentRecommender(BaseAgentRecommender):
    """Pre-Agent Recommender based on Neural Collaborative Filtering"""

    def __init__(self, neural_cf_model=None):
        super().__init__()
        self.neural_cf_model = neural_cf_model

    def _load_models(self) -> None:
        """
        It loads model or models needed for recommendations and raise
         exception if it is not available in the database.
        """

        if self.neural_cf_model is None:
            try:
                self.neural_cf_model = load_last_module(NEURAL_CF)
            except NoSavedModuleError as no_saved_module:
                raise UntrainedPreAgentError from no_saved_module

    def _for_logged_user(
        self, user: User, search_data: SearchData, k: int
    ) -> List[int]:
        """
        Generate recommendation for logged user.

        Args:
            user: user for whom recommendation will be generated.
            search_data: search phrase and filters information for narrowing
             down an item space.

        Returns:
            List of recommended services ids.
        """

        accessed_services_ids = [s.id for s in user.accessed_services]
        candidate_services = list(retrieve_services(search_data, accessed_services_ids))
        recommended_services = self._fill_candidate_services(
            candidate_services, k, accessed_services_ids
        )
        recommended_services_ids = [s.id for s in recommended_services]

        (
            users_ids,
            users_tensor,
            services_ids,
            services_tensor,
        ) = user_and_services_to_tensors(user, recommended_services)

        matching_probs = self.neural_cf_model(
            users_ids, users_tensor, services_ids, services_tensor
        )
        matching_probs = torch.reshape(matching_probs, (-1,)).tolist()
        top_k = sorted(
            list(zip(matching_probs, recommended_services_ids)), reverse=True
        )[:k]

        return [pair[1] for pair in top_k]
