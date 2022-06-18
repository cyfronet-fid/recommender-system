# pylint: disable=unused-argument, too-few-public-methods, missing-class-docstring, missing-function-docstring
# pylint: disable=no-self-use, superfluous-parens, no-else-return, no-member, not-callable, line-too-long

"""Neural Collaborative Filtering Inference Component"""

from typing import List, Tuple

import torch

from recommender.engines.base.base_inference_component import MLEngineInferenceComponent
from recommender.engines.ncf.ml_components.neural_collaborative_filtering import (
    NeuralCollaborativeFilteringModel,
    NEURAL_CF,
)
from recommender.engines.nlp_embedders.embedders import (
    Services2tensorsEmbedder,
    Users2tensorsEmbedder,
)
from recommender.errors import (
    InsufficientRecommendationSpaceError,
)
from recommender.models import User, SearchData
from recommender.services.fts import retrieve_services_for_recommendation
from logger_config import get_logger

logger = get_logger(__name__)


class NCFInferenceComponent(MLEngineInferenceComponent):
    """
    Recommender engine that provides logged-in users with personalized recommendations in a given context.
    """

    engine_name = "NCF"

    def __init__(self, K: int) -> None:
        self.neural_cf_model = None
        super().__init__(K)
        self.services2tensors_embedder = Services2tensorsEmbedder()
        self.users2tensors_embedder = Users2tensorsEmbedder()

    def _load_models(self) -> None:
        """It loads model or models needed for recommendations and raise
        exception if it is not available in the database.
        """

        self.neural_cf_model = NeuralCollaborativeFilteringModel.load(version=NEURAL_CF)
        self.neural_cf_model.eval()

    def user_and_services_to_tensors(self, user, services):
        """Used for inferention in recommendation endpoint.
        It takes raw MongoEngine models of one user and related services
        and compose them into tensors ready for inference.
        """

        services = list(services)
        services_ids = torch.tensor([service.id for service in services])
        services_tensor, _ = self.services2tensors_embedder(services)

        services_t_shape = services_tensor.shape[0]
        users_ids = torch.full([services_t_shape], user.id)
        users_tensor = self.users2tensors_embedder([user])[0].repeat(
            services_t_shape, 1
        )

        return users_ids, users_tensor, services_ids, services_tensor

    def _generate_recommendations(
        self, user: User, elastic_services: Tuple[int], search_data: SearchData
    ) -> List[int]:
        """Generate recommendation for logged user.

        Args:
            user: user for whom recommendation will be generated.
            elastic_services: item space from the Marketplace.
            search_data: search phrase and filters information for narrowing
             down an item space.

        Returns:
            recommended_services_ids: List of recommended services ids.
        """

        candidate_services = list(
            retrieve_services_for_recommendation(
                elastic_services, user.accessed_services
            )
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
        top_k = sorted(list(zip(matching_probs, candidate_services_ids)), reverse=True)[
            : self.K
        ]

        recommended_services_ids = [pair[1] for pair in top_k]

        return recommended_services_ids
