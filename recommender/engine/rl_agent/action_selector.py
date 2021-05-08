# pylint: disable=missing-module-docstring, invalid-name, no-member, too-few-public-methods

from typing import Dict, Any, Optional, Tuple

import pandas as pd
import torch
import torch.nn.functional as F

from torch import nn

from recommender.engine.pre_agent.models import load_last_module, SERVICES_AUTOENCODER
from recommender.errors import InsufficientRecommendationSpace
from recommender.models import User, Service
from recommender.services.fts import (
    retrieve_forbidden_service_ids,
    retrieve_matching_service_ids,
)
from recommender.services.services_history_generator import get_ordered_services


class ActionSelector:
    """Responsible for strategy and selection of services to
    recommend, given output of the Actor"""

    def __init__(self, K: int, service_embedder: Optional[nn.Module] = None) -> None:
        """
        Args:
            K: number of recommended services
            service_embedder: encoder part of the Service AutoEncoder
        """
        self.K = K
        self.itemspace = self._create_itemspace(
            service_embedder or load_last_module(SERVICES_AUTOENCODER).encoder
        )
        self.forbidden_indices = self._get_service_indices(
            retrieve_forbidden_service_ids()
        )
        if self.I - len(self.forbidden_indices) < self.K:
            raise InsufficientRecommendationSpace

    def __call__(
        self, weights: torch.Tensor, user: User, search_data: Dict[str, Any]
    ) -> Tuple[int]:
        """
        Based on weights_tensor, user and search_data, it selects services for
         recommendation and create action out of them.

        Args:
            weights: Weights returned by an actor model, tensor of shape [K, SE], where:
                - K is the number of services required for the recommendation
                - SE is a service content tensor embedding dim
            user: User for whom the recommendation is generated.
                User's accessed_services is used for narrowing the itemspace down.
            search_data: Information used for narrowing the itemspace down

        Returns:
            The tuple of recommended services ids
        """

        search_data_indices = self._get_service_indices(
            retrieve_matching_service_ids(search_data)
        )
        user_accessed_indices = self._get_service_indices(
            list(map(lambda s: s.id, get_ordered_services(user)))
        )

        mask = torch.zeros(self.I)
        mask[search_data_indices] = 1
        mask[user_accessed_indices] = torch.rand(()).item()
        mask[self.forbidden_indices] = -1

        engagement_values = F.softmax(weights @ self.itemspace.T, dim=1)
        recommended_indices = self._choose_recommended_indices(engagement_values, mask)

        return self.index_id_map.iloc[recommended_indices].id.values.tolist()

    def _choose_recommended_indices(self, engagement_values, mask):
        masked_engagement_values = mask * engagement_values
        top_K_indices = (
            torch.topk(masked_engagement_values, self.K).indices.numpy().tolist()
        )
        indices = [top_K_indices[0][0]]

        for k in range(1, self.K):
            indices += [next(filter(lambda x: x not in indices, top_K_indices[k]))]

        return indices

    def _get_service_indices(self, ids):
        return self.index_id_map[self.index_id_map.id.isin(ids)].index.values

    def _create_itemspace(self, services_embedder) -> torch.Tensor:
        """
        Creates itemspace tensor.

        Returns:
            itemspace: tensor of shape [I, SE] where: SE is service content tensor
             embedding dim and I is the number of services
        """

        all_services = Service.objects.order_by("id")
        self.I = len(all_services)
        self.index_id_map = pd.DataFrame(
            sorted(all_services.distinct("id")), columns=["id"]
        )

        service_one_hot_tensors = torch.Tensor([s.tensor for s in all_services])

        with torch.no_grad():
            service_embedded_tensors = services_embedder(service_one_hot_tensors)

        return service_embedded_tensors
