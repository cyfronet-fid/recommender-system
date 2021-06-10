# pylint: disable=missing-module-docstring, invalid-name, no-member, too-few-public-methods
from typing import List

import torch

from recommender.engine.agents.rl_agent.utils import (
    get_service_indices,
    create_itemspace,
)
from recommender.engine.models.autoencoders import create_embedder, SERVICES_AUTOENCODER
from recommender.engine.utils import load_last_module, NoSavedModuleError
from recommender.errors import MissingComponentError


class ActionInverter:
    """This class converts action as ids of K services to
    full weight matrix (of shape [K, SE] as returned by actor.
    It should be used when creating a dataset for RL agent"""

    def __init__(self, service_embedder=None):
        self.service_embedder = service_embedder

        self._load_components()

        self.itemspace, self.index_id_map = create_itemspace(self.service_embedder)
        self.itemspace_inverse = self.itemspace.pinverse().T
        self.itemspace_size = self.itemspace.shape[0]

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
            scoring_matrix = torch.rand(K, self.itemspace_size)

            for k, chosen_index in enumerate(recommended_indices):
                scoring_matrix[k][chosen_index] = 1.0

            decoded_weights = scoring_matrix @ self.itemspace_inverse
            decoded_weights_batch.append(decoded_weights)

        decoded_weights_batch = torch.stack(decoded_weights_batch, dim=0)

        return decoded_weights_batch

    def _load_components(self):
        try:
            self.service_embedder = self.service_embedder or create_embedder(
                load_last_module(SERVICES_AUTOENCODER)
            )
        except NoSavedModuleError as no_saved_module:
            raise MissingComponentError from no_saved_module

        self.service_embedder.eval()
