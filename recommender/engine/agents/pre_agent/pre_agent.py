# pylint: disable=unused-argument, too-few-public-methods, missing-class-docstring, missing-function-docstring
# pylint: disable=no-self-use, superfluous-parens, no-else-return, fixme, no-member

"""Implementation of the recommender engine Pre-Agent"""

from typing import List, Optional

import torch

from recommender.models import SearchData
from recommender.engine.agents.base_agent import BaseAgent
from recommender.errors import MissingComponentError, InsufficientRecommendationSpace
from recommender.engine.agents.pre_agent.models.neural_colaborative_filtering import (
    NEURAL_CF,
    NeuralColaborativeFilteringModel,
)
from recommender.engine.preprocessing import (
    user_and_services_to_tensors,
)
from recommender.models import User

from recommender.engine.utils import NoSavedModuleError, load_last_module
from recommender.services.fts import retrieve_services_for_recommendation

PRE_AGENT = "pre_agent"


class PreAgent(BaseAgent):
    """Pre-Agent Recommender based on Neural Collaborative Filtering"""

    def __init__(
        self, neural_cf_model: Optional[NeuralColaborativeFilteringModel] = None
    ) -> None:
        self.neural_cf_model = neural_cf_model
        super().__init__()

    def _load_models(self) -> None:
        """
        It loads model or models needed for recommendations and raise
         exception if it is not available in the database.
        """

        if self.neural_cf_model is None:
            try:
                self.neural_cf_model = load_last_module(NEURAL_CF)
            except NoSavedModuleError as no_saved_module:
                raise MissingComponentError from no_saved_module

    def _for_logged_user(
        self, user: User, search_data: SearchData, K: int
    ) -> List[int]:
        """
        Generate recommendation for logged user.

        Args:
            user: user for whom recommendation will be generated.
            search_data: search phrase and filters information for narrowing
             down an item space.

        Returns:
            recommended_services_ids: List of recommended services ids.
        """

        candidate_services = list(
            retrieve_services_for_recommendation(search_data, user.accessed_services)
        )
        if len(candidate_services) < K:
            raise InsufficientRecommendationSpace
        candidate_services_ids = [s.id for s in candidate_services]

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
        top_k = sorted(list(zip(matching_probs, candidate_services_ids)), reverse=True)[
            :K
        ]

        recommended_services_ids = [pair[1] for pair in top_k]

        return recommended_services_ids
