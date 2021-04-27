from typing import List

import torch

from models import SearchData


class FiltersEmbedder:
    def __init__(self):
        pass

    def __call__(self, search_data: SearchData) -> torch.Tensor:
        """
        Embedd filters using some embbeding

        Args:
            search_data: Search data with filters.

        Returns:
            Tensor of the embedded filters.
        """

        # TODO: implement
        pass
