# pylint: disable=missing-module-docstring, invalid-name, no-member, too-few-public-methods, no-self-use

from typing import Optional, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from recommender.engine.agents.rl_agent.utils import use_service_embedder
from recommender.engine.models.autoencoders import SERVICES_AUTOENCODER, create_embedder
from recommender.engine.utils import load_last_module, NoSavedModuleError
from recommender.errors import InsufficientRecommendationSpace, MissingComponentError
from recommender.models import Service


class ServiceSelector:
    """Responsible for strategy and selection of services to
    recommend, given output of the Actor"""

    def __init__(self, service_embedder: Optional[nn.Module] = None) -> None:
        """
        Args:
            service_embedder: encoder part of the Service AutoEncoder
        """

        self.service_embedder = service_embedder

        self._load_components()

        self.itemspace, self.index_id_map = self._create_itemspace()
        self.itemspace_size = self.itemspace.shape[0]

    def __call__(
        self,
        K: int,
        weights: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[int]:
        """
        Based on weights_tensor, user and search_data, it selects services for
         recommendation and returns them

        Args:
            K: number of recommended services
            weights: Weights returned by an actor model, tensor of shape [K, SE], where:
                - K is the number of services required for the recommendation
                - SE is a service content tensor embedding dim

        Returns:
            The tuple of recommended services ids
        """

        if (mask > 0).sum() < K:
            raise InsufficientRecommendationSpace

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

    def _create_itemspace(self) -> Tuple[torch.Tensor, pd.DataFrame]:
        all_services = list(Service.objects.order_by("id"))
        service_embedded_tensors, index_id_map = use_service_embedder(
            all_services, self.service_embedder
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
