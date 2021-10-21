# pylint: disable=line-too-long
"""Autoencoder Data Extraction Step."""

from recommender.engines.base.base_steps import DataExtractionStep
from recommender.models import User, Service

AUTOENCODERS = "autoencoders"
USERS = "users"
NUM_OF_USERS = "num_of_users"
SERVICES = "services"
NUM_OF_SERVICES = "num_of_services"
DETAILS = "details"


def get_users_and_services():
    """Get users and services from the database"""
    users = User.objects.order_by("-id")
    services = Service.objects.order_by("-id")

    num_of_usr = users.count()
    num_of_srv = services.count()

    return list(users), list(services), num_of_usr, num_of_srv


def count_details(num_of_usr: int, num_of_srv: int):
    """Get how many users and services there are in the self.data"""

    details = {USERS: {}, SERVICES: {}}

    details[USERS][NUM_OF_USERS] = num_of_usr
    details[SERVICES][NUM_OF_SERVICES] = num_of_srv

    return details


class AEDataExtractionStep(DataExtractionStep):
    """Autoencoder data extraction step"""

    def __init__(self, pipeline_config):
        super().__init__(pipeline_config)
        self.data = {AUTOENCODERS: {USERS: {}, SERVICES: {}}}

    def __call__(self, data=None):
        users, services, num_of_usr, num_of_srv = get_users_and_services()

        self.data[AUTOENCODERS][USERS] = users
        self.data[AUTOENCODERS][SERVICES] = services

        details = count_details(num_of_usr, num_of_srv)

        return self.data, details
