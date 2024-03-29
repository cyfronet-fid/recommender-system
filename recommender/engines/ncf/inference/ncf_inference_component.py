# pylint: disable=unused-argument, too-few-public-methods, missing-class-docstring, missing-function-docstring
# pylint: disable=superfluous-parens, no-else-return, no-member, not-callable, line-too-long

"""Neural Collaborative Filtering Inference Component"""

from typing import List, Tuple

import torch

from recommender.engines.base.base_inference_component import MLEngineInferenceComponent
from recommender.engines.explanations import Explanation
from recommender.engines.ncf.ml_components.neural_collaborative_filtering import (
    NeuralCollaborativeFilteringModel,
    NEURAL_CF,
)
from recommender.errors import (
    InsufficientRecommendationSpaceError,
    NoPrecalculatedTensorsError,
)
from recommender.models import User, SearchData
from recommender.services.fts import retrieve_services_for_recommendation
from logger_config import get_logger

logger = get_logger(__name__)


class NCFInferenceComponent(MLEngineInferenceComponent):
    """
    Recommender engine that provides logged-in users with personalized
     recommendations in a given context.
    """

    engine_name = "NCF"
    default_explanation = Explanation(
        long="This service has been selected based on historical orders of the"
        " given user and similar orders of similar users",
        short="This service has been selected based on historical orders.",
    )

    def __init__(self, K: int) -> None:
        self.neural_cf_model = None
        super().__init__(K)

    def _load_models(self) -> None:
        """It loads model or models needed for recommendations and raise
        exception if it is not available in the database.
        """

        self.neural_cf_model = NeuralCollaborativeFilteringModel.load(version=NEURAL_CF)
        self.neural_cf_model.eval()

    @staticmethod
    def user_and_services_to_tensors(user, services):
        """Used for inferention in recommendation endpoint.
        It takes raw MongoEngine models of one user and related services
        and compose them into tensors ready for inference.
        """

        if not user.dense_tensor:
            raise NoPrecalculatedTensorsError("Given user has no precalculated dense")

        for service in services:
            if not service.dense_tensor:
                raise NoPrecalculatedTensorsError(
                    "One or more of given services has/have no precalculated"
                    " dense tensor(s)"
                )

        services_ids = []
        services_tensors = []
        services = list(services)
        for service in services:
            services_ids.append(service.id)
            service_tensor = torch.unsqueeze(torch.Tensor(service.dense_tensor), dim=0)
            services_tensors.append(service_tensor)

        services_ids = torch.tensor(services_ids)
        services_tensor = torch.cat(services_tensors, dim=0)

        services_t_shape = services_tensor.shape[0]
        users_ids = torch.full([services_t_shape], user.id)
        user_tensor = torch.Tensor(user.dense_tensor)
        users_tensor = torch.unsqueeze(user_tensor, dim=0).repeat(services_t_shape, 1)

        return users_ids, users_tensor, services_ids, services_tensor

    def _get_ranking(
        self, user: User, candidates: Tuple[int], search_data: SearchData
    ) -> Tuple[List[int], List[float]]:
        """Generate services ranking.

        Args:
            user: user for whom recommendation will be generated.
            candidates: item space from the Marketplace.
            search_data: search phrase and filters information for narrowing
             down an item space.

        Returns:
            ranking: Ranking of services.
        """

        candidate_services = list(
            retrieve_services_for_recommendation(candidates, user.accessed_services)
        )
        if len(candidate_services) < self.K:
            raise InsufficientRecommendationSpaceError()
        candidate_services_ids = [s.id for s in candidate_services]

        (
            users_ids,
            users_tensor,
            services_ids,
            services_tensor,
        ) = self.user_and_services_to_tensors(user, candidate_services)

        matching_probs = self.neural_cf_model(
            users_ids, users_tensor, services_ids, services_tensor
        )
        matching_probs = torch.reshape(matching_probs, (-1,)).tolist()
        ranking = sorted(
            list(zip(matching_probs, candidate_services_ids)), reverse=True
        )

        scores, recommended_services_ids = [list(t) for t in list(zip(*ranking))]

        return recommended_services_ids, scores

    def _generate_recommendations(
        self, user: User, candidates: Tuple[int], search_data: SearchData
    ) -> Tuple[List[int], List[float], List[Explanation]]:
        """Generate recommendation for logged user.

        Args:
            user: user for whom recommendation will be generated.
            candidates: item space from the Marketplace.
            search_data: search phrase and filters information for narrowing
             down an item space.

        Returns:
            recommended_services_ids: List of recommended services ids.
            scores: List of ranking scores for all recommended services.
            explanations: List of explanations for all recommended services.
        """

        recommended_services_ids, scores = tuple(
            l[: self.K] for l in self._get_ranking(user, candidates, search_data)
        )
        explanations = self._generate_explanations()

        return recommended_services_ids, scores, explanations
