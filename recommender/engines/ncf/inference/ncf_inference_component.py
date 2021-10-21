# pylint: disable=unused-argument, too-few-public-methods, missing-class-docstring, missing-function-docstring
# pylint: disable=no-self-use, superfluous-parens, no-else-return, no-member

"""Neural Collaborative Filtering Inference Component"""

from typing import List

import torch

from recommender.engines.autoencoders.training.data_preparation_step import (
    user_and_services_to_tensors,
)
from recommender.engines.base.base_inference_component import BaseInferenceComponent
from recommender.engines.ncf.ml_components.neural_collaborative_filtering import (
    NeuralCollaborativeFilteringModel,
    NEURAL_CF,
)
from recommender.errors import InsufficientRecommendationSpace
from recommender.models import User, SearchData
from recommender.services.fts import retrieve_services_for_recommendation


class NCFInferenceComponent(BaseInferenceComponent):
    """Neural Collaborative Filtering Inference Component"""

    def __init__(self, K: int) -> None:
        self.neural_cf_model = None
        super().__init__(K)

    def _load_models(self) -> None:
        """It loads model or models needed for recommendations and raise
        exception if it is not available in the database.
        """

        self.neural_cf_model = NeuralCollaborativeFilteringModel.load(version=NEURAL_CF)
        self.neural_cf_model.eval()

    def _for_logged_user(self, user: User, search_data: SearchData) -> List[int]:
        """Generate recommendation for logged user.

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
        if len(candidate_services) < self.K:
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
            : self.K
        ]

        recommended_services_ids = [pair[1] for pair in top_k]

        return recommended_services_ids
