from typing import Tuple

import torch

from models import User, SearchData


class ActionSelector:
    def __init__(self) -> None:
        self.itemspace = self._create_itemspace()

    def __call__(self, weights: torch.Tensor, user: User, search_data: SearchData) -> Tuple[int]:
        """
        Based on weights_tensor, user and search_data, it selects services for
         recommendation and create action out of them.

        Args:
            weights: Weights returned by an actor model, tensor of shape [K, SE], where:
                - K is the number of services required for the recommendation
                - SE is a service content tensor embedding dim
            user: User for whom the recommendation is generated. User's accessed_services is used for narrowing the itemspace down.
            search_data: Information used for narrowing the itemspace down

        Returns:
            The tuple of recommended services ids
        """

        # TODO: implement (performance is crucial)
        pass

    def _create_itemspace(self) -> torch.Tensor:
        """
        Creates itemspace tensor.

        Returns:
            itemspace: tensor of shape [SE, I] where: SE is service content tensor
             embedding dim and I is the number of services
        """

        # TODO: implement
        pass
