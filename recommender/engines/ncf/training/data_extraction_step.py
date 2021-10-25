"""Neural Collaborative Filtering Data Extraction Step"""

import random
from copy import deepcopy
from typing import Tuple, Dict, Union, Iterable, List

import numpy as np
from mongoengine import QuerySet
from tqdm.auto import tqdm

from recommender.engines.base.base_steps import DataExtractionStep
from recommender.engines.constants import VERBOSE
from recommender.models import Service, User

USER = "user"
ORDERED_SERVICES = "ordered_services"
NOT_ORDERED_SERVICES = "not_ordered_services"

MAX_USERS = "max_users"

RAW_DATA = "raw_data"
USERS_MAX_ID = "users_max_id"
SERVICES_MAX_ID = "services_max_id"

ZEROS = "zeros"
MIN = "min"
MAX = "max"
MEAN = "mean"

STATISTICS = "statistics"

USERS = "users"
SERVICES = "services"


def _get_not_ordered_services(ordered_services: Union[Iterable, QuerySet]) -> List:
    """Given ordered services find not ordered services.

    Args:
        ordered_services: Iterable of ordered services.

    Returns:
        not_ordered_services: List of not ordered services that has the same
         length as ordered_services.
    """

    ordered_services = list(ordered_services)
    ordered_services_ids = [s.id for s in ordered_services]
    all_not_ordered_services = list(Service.objects(id__nin=ordered_services_ids))
    k = min(len(ordered_services), len(all_not_ordered_services))
    not_ordered_services = random.sample(all_not_ordered_services, k=k)

    return not_ordered_services


def calc_statistics(raw_data: List[Dict[str, Dict]]) -> Dict[str, Union[int, float]]:
    """Calculate basic statistics for the given raw_data.

    Args:
        raw_data: Raw data produced in the NCFDataExtractionStep.__call__.

    Returns:
        statistics: basic statistics of the raw_data.
    """
    ordered_services_numbers = []
    for entry in raw_data:
        ordered_services_numbers.append(len(entry[ORDERED_SERVICES]))

    zeros = len([n for n in ordered_services_numbers if n == 0])
    os_min = min(ordered_services_numbers)
    os_max = max(ordered_services_numbers)
    os_mean = np.mean(ordered_services_numbers)

    statistics = {ZEROS: zeros, MIN: os_min, MAX: os_max, MEAN: os_mean}

    return statistics


class NCFDataExtractionStep(DataExtractionStep):
    """Neural Collaborative Filtering Data Extraction Step"""

    def __init__(self, pipeline_config):
        super().__init__(pipeline_config)

        self.max_users = self.resolve_constant(MAX_USERS)
        self.verbose = self.resolve_constant(VERBOSE, False)

    def __call__(self, data: Dict = None) -> Tuple[Dict, Dict]:
        """Creates balanced raw dataset that consist of pairs user-service.

        If there is n users and each of them has on average K services
        then the final dataset will consist of 2kn examples
        (not just kn because for each K positive examples of services
        oredered by a user there are generated also K negative services
        not ordered by a user).

        Time and space complexity of this algorithm is O(kn)
        """

        raw_data = []

        users = list(User.objects)
        if self.max_users is not None:
            users = users[: self.max_users]

        for user in tqdm(
            users, desc="Generating dataset...", disable=(not self.verbose)
        ):
            ordered_services = user.accessed_services
            not_ordered_services = deepcopy(
                _get_not_ordered_services(ordered_services)
            )  # (same amount as positive - classes balance)

            raw_data.append(
                {
                    USER: user,
                    ORDERED_SERVICES: ordered_services,
                    NOT_ORDERED_SERVICES: not_ordered_services,
                }
            )

        details = {STATISTICS: calc_statistics(raw_data)}

        users_max_id = User.objects.order_by("-id").first().id
        services_max_id = Service.objects.order_by("-id").first().id

        data = {
            RAW_DATA: raw_data,
            USERS_MAX_ID: users_max_id,
            SERVICES_MAX_ID: services_max_id,
            USERS: User.objects,
            SERVICES: Service.objects,
        }

        return data, details
