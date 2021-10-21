# pylint: disable=missing-module-docstring, invalid-name, no-member, too-few-public-methods, fixme
from typing import List

import torch

from recommender.engine.agents.rl_agent.utils import (
    get_service_indices,
)
from recommender.engines.autoencoders.ml_components.embedder import Embedder
from recommender.models import Service


class Services2Weights:
    """This class converts list of services (of size K)
    represented as their DB ids to
    full weight matrix (of shape [K, SE] as returned by actor. It works on batches.
    It should be used when creating a dataset for RL agent"""

    def __init__(self, service_embedder: Embedder):
        all_services = list(Service.objects.order_by("id"))
        self.itemspace, self.index_id_map = service_embedder(all_services)
        self.itemspace_inverse = self.itemspace.pinverse().T
        self.I = self.itemspace.shape[0]

    def __call__(self, recommended_id_batch: List[List[int]]) -> torch.Tensor:
        """
        Args:
            recommended_id_batch: batches of DB ids of recommended services
        Returns:
            decoded_weights_batch: tensor of weights batches compatible with actor model
        """

        # TODO: better batch handling (optimization)
        decoded_weights_batch = []
        for recommended_ids in recommended_id_batch:
            recommended_indices = get_service_indices(
                self.index_id_map, recommended_ids
            )
            K = len(recommended_indices)
            scoring_matrix = torch.rand(K, self.I)

            for k, chosen_index in enumerate(recommended_indices):
                scoring_matrix[k][chosen_index] = 1.0

            decoded_weights = scoring_matrix @ self.itemspace_inverse
            decoded_weights_batch.append(decoded_weights)

        decoded_weights_batch = torch.stack(decoded_weights_batch, dim=0)

        return decoded_weights_batch
