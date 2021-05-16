# pylint: disable=invalid-name

"""Critic Model implementation"""

from typing import Tuple

import torch


class Critic(torch.nn.Module):
    """Critic Model"""

    def __init__(self, K: int, SE: int, history_embedder: torch.nn.Module):
        super().__init__()
        self.K = K
        self.SE = SE
        self.history_embedder = history_embedder
        # WARNING: history_embedder is a model shared between actor and critic,
        # the .detach will be probably needed for proper training

        # TODO: layers initialization
        self.output_layer = torch.nn.Linear(None, 1)

    def forward(
        self, state: Tuple[torch.Tensor], action: Tuple[torch.Tensor]
    ) -> torch.Tensor:
        """
        Performs forward propagation.

        Args:
            state: Tuple of following tensors:
                - user: Embedded user content tensor of shape [UE]
                - services_history: Services history tensor of shape [N, SE]
                - filters: Embedded filters tensor of shape [FE]
                - search_phrase: Embedded search phrase tensor of shape [SPE]
                where:
                  - UE is user content tensor embedding dim
                  - N is user clicked services history length
                  - SE is service content tensor embedding dim
                  - FE is filters tensor embedding dim
                  - SPE is search phrase tensor embedding dim'
            action: TODO

        Returns:
            The value of taking the given action at the given state. It's a
             tensor of shape [] (scalar, but has be torch.Tensor to have
             backpropagation capabilities)
        """

        user, services_history, filters, search_phrase = state
        services_history = self.history_embedder(services_history)

        # action is a tuple of K service content tensors
        # They are not concatenated by Action Embedder because critic may want
        # to use each of them separately in some kind of architecture, for
        # example each tensor can be feeded into the same submodule and then
        # results can be concatenated

        # TODO: implement forward computation
        action_value = self.output_layer(None)
        return action_value
