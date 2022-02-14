# pylint: disable=missing-module-docstring, no-member, too-few-public-methods, invalid-name
from typing import List

import torch

from recommender.engines.rl.utils import (
    create_index_id_map,
    get_service_indices,
)
from recommender.models import SearchData, User, Service
from recommender.services.fts import retrieve_forbidden_services, filter_services
from recommender.engines.rl.ml_components.services_history_generator import (
    get_ordered_services,
)
from recommender.errors import SizeOfUsersAndSearchDataError


class SearchDataEncoder:
    """Encodes search data and user's ordered services into a binary mask"""

    def __init__(self):
        all_services = list(Service.objects.order_by("id"))
        self.I = len(all_services)
        self.index_id_map = create_index_id_map(all_services)

        forbidden_services = retrieve_forbidden_services()
        self.forbidden_service_indices = get_service_indices(
            self.index_id_map, forbidden_services.distinct("id")
        )

    def __call__(
        self, users: List[User], search_data: List[SearchData]
    ) -> torch.Tensor:
        if not len(users) == len(search_data):
            raise SizeOfUsersAndSearchDataError()

        batch_size = len(users)

        mask = torch.zeros(batch_size, self.I)

        for i in range(batch_size):
            filtered_services = filter_services(search_data[i])
            ordered_services = get_ordered_services(users[i])

            filtered_service_indices = get_service_indices(
                self.index_id_map, filtered_services.distinct("id")
            )

            ordered_service_indices = get_service_indices(
                self.index_id_map, [s.id for s in ordered_services]
            )

            mask[i, filtered_service_indices] = 1
            mask[i, ordered_service_indices] = 0
            mask[i, self.forbidden_service_indices] = 0

        return mask
