from typing import Tuple

import torch

from models import Service


class ActionEmbedder:
    def __init__(self, service_embedder=None):
        self.service_embedder = service_embedder

    def __call__(self, action: Tuple[Service]) -> Tuple[torch.Tensor]:
        """
        Embedd each service of the action using Service Embedder.

        Args:
            action: Tuple of services.

        Returns:
            Tuple of service embedded content tensors of shape [SE]
        """

        # TODO: implement
        pass