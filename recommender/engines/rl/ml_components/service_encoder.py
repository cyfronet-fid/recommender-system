# pylint: disable=missing-module-docstring, no-member, too-few-public-methods, invalid-name, line-too-long
from typing import List
from copy import deepcopy
import torch
from tqdm import tqdm
from recommender.engines.rl.utils import (
    create_index_id_map,
    get_service_indices,
)
from recommender.models import User, Service, State
from recommender.services.fts import retrieve_forbidden_services
from recommender.engines.rl.ml_components.services_history_generator import (
    get_ordered_services,
)
from recommender.errors import SizeOfUsersAndElasticServicesError
from logger_config import get_logger

logger = get_logger(__name__)


class ServiceEncoder:
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
        self, users: List[User], states: List[State], verbose=False
    ) -> torch.Tensor:
        batch_size = len(users)
        states_len = len(states)

        if batch_size != states_len:
            # Each state has elastic_services
            raise SizeOfUsersAndElasticServicesError

        mask = torch.zeros(batch_size, self.I)

        if verbose:
            logger.info("Creating mask...")

        for i in tqdm(range(batch_size), desc="States", disable=not verbose):
            state = states.pop(0)

            if states_len == 1:
                # Model evaluation step - searchdata there is not saved, so it cannot be accessed
                # This avoids mongodb ValidationError
                state.search_data = None

            # Just reading the tuple of references (which is the elastic_services)
            # causes the IDs to be replaced with entire service objects in all referenced tuple.
            # To avoid memory leak:
            # - state has 3 references - direct manipulation on state object will result in memory leak
            #   to avoid that, there is deepcopy made
            state = deepcopy(state)
            elastic_services = state.elastic_services
            services_context = self.get_id_from_services(elastic_services)

            ordered_services = get_ordered_services(users[i])

            services_context_indices = get_service_indices(
                self.index_id_map, services_context
            )
            ordered_service_indices = get_service_indices(
                self.index_id_map, [s.id for s in ordered_services]
            )

            mask[i, services_context_indices] = 1
            mask[i, ordered_service_indices] = 0
            mask[i, self.forbidden_service_indices] = 0

        return mask

    @staticmethod
    def get_id_from_services(services: List[Service]) -> List[int]:
        """Get services IDs"""
        return [service.id for service in services]
