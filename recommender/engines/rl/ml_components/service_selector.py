# pylint: disable=missing-module-docstring, invalid-name, no-member, too-few-public-methods

from typing import Tuple, List

import torch
import torch.nn.functional as F

from recommender.engines.autoencoders.ml_components.embedder import Embedder
from recommender.errors import InsufficientRecommendationSpaceError
from recommender.models import Service
from logger_config import get_logger

logger = get_logger(__name__)


class ServiceSelector:
    """Responsible for strategy and selection of services to
    recommend, given output of the Actor"""

    def __init__(
        self,
        service_embedder: Embedder,
        use_cached_embeddings: bool = True,
        save_cached_embeddings: bool = False,
    ) -> None:
        """
        Args:
            service_embedder: Embedder of the services
            use_cached_embeddings: Flag, describing if we should use cached services'
                and users' embeddings
            save_cached_embeddings: Flag, describing if we should
                cache computed services' and users' embeddings
        """
        all_services = list(Service.objects.order_by("id"))
        self.itemspace, self.index_id_map = service_embedder(
            all_services,
            use_cache=use_cached_embeddings,
            save_cache=save_cached_embeddings,
        )

    def __call__(
        self,
        weights: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[List[int], List[float]]:
        """
        Based on weights_tensor, user and search_data, it selects services for
         recommendation and returns them

        Args:
            weights: Weights returned by an actor model, tensor of shape [K, SE], where:
                - K is the number of services required for the recommendation
                - SE is a service content tensor embedding dim

        Returns:
            recommended_services_ids: List of recommended services ids.
            scores: List of ranking scores for all recommended services.
        """

        K = weights.shape[0]

        if (mask > 0).sum() < K:
            raise InsufficientRecommendationSpaceError()

        engagement_values = F.softmax(weights @ self.itemspace.T, dim=1)

        recommended_indices, scores = self._choose_recommended_indices(
            engagement_values, mask, K
        )

        recommended_services_ids = self.index_id_map.iloc[
            recommended_indices
        ].id.values.tolist()

        return recommended_services_ids, scores

    def _choose_recommended_indices(self, engagement_values, mask, K):
        masked_engagement_values = mask * engagement_values
        top_K_indices = torch.topk(masked_engagement_values, K).indices.numpy().tolist()
        indices = [top_K_indices[0][0]]
        for k in range(1, K):
            indices += [next(filter(lambda x: x not in indices, top_K_indices[k]))]
        scores = [masked_engagement_values[k][indices[k]] for k in range(K)]

        return indices, scores
