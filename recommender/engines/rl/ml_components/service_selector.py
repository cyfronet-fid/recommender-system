# pylint: disable=missing-module-docstring, invalid-name, no-member, too-few-public-methods

from typing import Tuple

import torch
import torch.nn.functional as F

from recommender.engines.nlp_embedders.embedders import Services2tensorsEmbedder
from recommender.errors import InsufficientRecommendationSpaceError
from recommender.models import Service
from logger_config import get_logger

logger = get_logger(__name__)


class ServiceSelector:
    """Responsible for strategy and selection of services to
    recommend, given output of the Actor"""

    def __init__(self) -> None:
        all_services = list(Service.objects.order_by("id"))
        self.itemspace, self.index_id_map = Services2tensorsEmbedder()(all_services)

    def __call__(
        self,
        weights: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[int]:
        """
        Based on weights_tensor, user and search_data, it selects services for
         recommendation and returns them

        Args:
            weights: Weights returned by an actor model, tensor of shape [K, SE], where:
                - K is the number of services required for the recommendation
                - SE is a service content tensor embedding dim

        Returns:
            The tuple of recommended services ids
        """

        K = weights.shape[0]

        if (mask > 0).sum() < K:
            raise InsufficientRecommendationSpaceError()

        engagement_values = F.softmax(weights @ self.itemspace.T, dim=1)

        recommended_indices = self._choose_recommended_indices(
            engagement_values, mask, K
        )

        return self.index_id_map.iloc[recommended_indices].id.values.tolist()

    def _choose_recommended_indices(self, engagement_values, mask, K):
        masked_engagement_values = mask * engagement_values
        top_K_indices = torch.topk(masked_engagement_values, K).indices.numpy().tolist()
        indices = [top_K_indices[0][0]]

        for k in range(1, K):
            indices += [next(filter(lambda x: x not in indices, top_K_indices[k]))]

        return indices
