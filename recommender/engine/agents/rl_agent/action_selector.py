# pylint: disable=missing-module-docstring, invalid-name, no-member, too-few-public-methods, no-self-use

from typing import Optional, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from recommender.engine.models.autoencoders import SERVICES_AUTOENCODER, create_embedder
from recommender.models import SearchData
from recommender.engine.agents.rl_agent.utils import (
    get_service_indices,
    use_service_embedder,
)
from recommender.engine.utils import load_last_module, NoSavedModuleError
from recommender.errors import InsufficientRecommendationSpace, MissingComponentError
from recommender.models import User, Service
from recommender.services.fts import (
    retrieve_forbidden_services,
    filter_services,
)
from recommender.services.services_history_generator import get_ordered_services


class ActionSelector:
    """Responsible for strategy and selection of services to
    recommend, given output of the Actor"""

    def __init__(self, service_embedder: Optional[nn.Module] = None) -> None:
        """
        Args:
            service_embedder: encoder part of the Service AutoEncoder
        """

        self.service_embedder = service_embedder

        self._load_components()

        self.itemspace, self.index_id_map = self._create_itemspace(
            self.service_embedder
        )
        self.itemspace_size = self.itemspace.shape[0]
        self.forbidden_indices = get_service_indices(
            self.index_id_map, retrieve_forbidden_services().only("id").distinct("id")
        )
        self.forbidden_services_size = len(self.forbidden_indices)

    def __call__(
        self, K: int, weights: torch.Tensor, user: User, search_data: SearchData
    ) -> Tuple[int]:
        """
        Based on weights_tensor, user and search_data, it selects services for
         recommendation and create action out of them.

        Args:
            K: number of recommended services
            weights: Weights returned by an actor model, tensor of shape [K, SE], where:
                - K is the number of services required for the recommendation
                - SE is a service content tensor embedding dim
            user: User for whom the recommendation is generated.
                User's ordered services is used for narrowing the itemspace down.
            search_data: Information used for narrowing the itemspace down

        Returns:
            The tuple of recommended services ids
        """

        if self.itemspace_size - self.forbidden_services_size < K:
            raise InsufficientRecommendationSpace

        search_data_indices = get_service_indices(
            self.index_id_map, filter_services(search_data).only("id").distinct("id")
        )
        user_ordered_indices = get_service_indices(
            self.index_id_map, list(map(lambda s: s.id, get_ordered_services(user)))
        )

        mask = torch.zeros(self.itemspace_size)
        mask[search_data_indices] = 1
        mask[user_ordered_indices] = torch.rand(()).item()
        mask[self.forbidden_indices] = -1

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

    def _create_itemspace(self, services_embedder) -> Tuple[torch.Tensor, pd.DataFrame]:
        all_services = Service.objects.order_by("id")
        service_embedded_tensors, index_id_map = use_service_embedder(
            all_services, services_embedder
        )

        return service_embedded_tensors, index_id_map

    def _load_components(self):
        try:
            self.service_embedder = self.service_embedder or create_embedder(
                load_last_module(SERVICES_AUTOENCODER)
            )
        except NoSavedModuleError as no_saved_module:
            raise MissingComponentError from no_saved_module

        self.service_embedder.eval()
