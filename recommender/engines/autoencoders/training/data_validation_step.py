# pylint: disable=line-too-long
"""Autoencoder Data Validation Step."""

from typing import Tuple, Dict

from recommender.engines.base.base_steps import DataValidationStep
from recommender.engines.autoencoders.training.data_extraction_step import AUTOENCODERS

from recommender.errors import (
    NotEnoughUsersOrServices,
    NoCategoriesScientificDomains,
)

DATA_IS_VALID = "data_is_valid"
LEAST_NUM_OF_USR_SRV = "least_num_of_usr_srv"


def valid_num_of_usr_and_srv(collection: list, least_num_of_usr_srv: int = 1):
    """Check if there are users and services in data"""
    if len(collection) < least_num_of_usr_srv:
        raise NotEnoughUsersOrServices()


def valid_properties(collection: list):
    """Check if users and services have at least one category and scientific domain in order to enable training"""
    if not any(
        len(col.categories) and len(col.scientific_domains) > 0 for col in collection
    ):
        raise NoCategoriesScientificDomains()


class AEDataValidationStep(DataValidationStep):
    """Autoencoder data validation step"""

    def __init__(self, pipeline_config):
        super().__init__(pipeline_config)
        self.least_num_of_usr_srv = self.resolve_constant(LEAST_NUM_OF_USR_SRV, 2)

    def __call__(self, data: Dict = None) -> Tuple[Dict, Dict]:
        details = {DATA_IS_VALID: False}
        raw_data = data[AUTOENCODERS].values()

        for collection in raw_data:
            valid_num_of_usr_and_srv(collection, self.least_num_of_usr_srv)
            valid_properties(collection)

        details[DATA_IS_VALID] = True

        return data, details
