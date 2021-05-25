# pylint: disable=too-few-public-methods, no-member

"""Implementation of the Action Encoder"""

from typing import Optional
from typing import List
import torch

from recommender.engine.models.autoencoders import SERVICES_AUTOENCODER, create_embedder
from recommender.engine.utils import load_last_module
from recommender.models import Service


class ActionEncoder:
    """Action Encoder"""

    def __init__(self, service_embedder: Optional[torch.nn.Module] = None):
        self.service_embedder = service_embedder or create_embedder(
            load_last_module(SERVICES_AUTOENCODER)
        )

    def __call__(self, action: List[Service]) -> torch.Tensor:
        """
        Encode each service of the action using Service Embedder.

        Args:
            action: List of services.

        Returns:
            Tensor of embedded services of shape [K, SE],
             where K is the number of services in action.
        """

        service_oh_tensors = torch.stack([torch.Tensor(s.tensor) for s in action])
        service_tensors = self.service_embedder(service_oh_tensors)

        return service_tensors
