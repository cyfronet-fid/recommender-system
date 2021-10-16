# pylint: disable=useless-super-delegation, invalid-name

"""Neural Collaborative Filtering Data Validation Step."""

from typing import Tuple, Dict, List, Union

from recommender.engines.base.base_steps import DataValidationStep
from recommender.engines.ncf.training.data_extraction_step import (
    ORDERED_SERVICES,
    NOT_ORDERED_SERVICES,
    RAW_DATA,
)
from recommender.errors import (
    TooFewOrderedServicesError,
    NoUserInDatasetError,
    ImbalancedDatasetError,
)
from recommender.models import Service, User

DATA_IS_VALID = "data_is_valid"
LEAST_N_ORDERS_PER_USER = "least_n_orders_per_user"


def exist_user_that_has_at_least_n_ordered_services(
    data: List[Dict[str, Union[User, List[Service]]]], n: int = 5
) -> None:
    """For small dataset typical proportions of examples in train, valid and
    test datasets are respectively: 0.6, 0.2, 0.2. This split will be
    applied to the (user, service) pair examples within each user and his
    services separately. Each dataset (train, test, valid) should have at
    least two examples. Hence, in the data list, should exist at least
    one user with at least 5 (default value) ordered services (the number of not ordered
    services is the same as ordered because classes are balanced - this is
    check in other assertion). For large datasets proportions of train,
    test, valid can be more like: 0.98, 0.01, 0.01, but still there should
    exist at least one user that have at least 10 ordered services -
    otherwise such a dataset can't be used for effective training.
    """

    for entry in data:
        if len(entry[ORDERED_SERVICES]) >= n:
            return
    raise TooFewOrderedServicesError()


def at_least_one_user(data) -> None:
    """Check if there is at least one user in the data."""

    if len(data) < 1:
        raise NoUserInDatasetError()


def classes_are_balanced(data):
    """Check if classes in the dataset are balanced."""

    for entry in data:
        if len(entry[ORDERED_SERVICES]) != len(entry[NOT_ORDERED_SERVICES]):
            raise ImbalancedDatasetError()


class NCFDataValidationStep(DataValidationStep):
    """Neural Collaborative Filtering Data Validation Step."""

    def __init__(self, pipeline_config):
        super().__init__(pipeline_config)
        self.n = self.resolve_constant(LEAST_N_ORDERS_PER_USER, 5)

    def __call__(self, data: Dict = None) -> Tuple[Dict, Dict]:
        """Perform data validation consisting of checking if:
        -> there is at least one user in the dataset,
        -> there is at least one user that has 5 oredered services,
        -> classes in the dataset are balanced.
        """

        details = {DATA_IS_VALID: False}

        raw_data = data[RAW_DATA]
        at_least_one_user(raw_data)
        exist_user_that_has_at_least_n_ordered_services(raw_data)
        classes_are_balanced(raw_data)

        details[DATA_IS_VALID] = True

        return data, details
