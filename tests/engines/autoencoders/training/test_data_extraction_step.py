# pylint: disable-all

from recommender.engines.autoencoders.training.data_extraction_step import (
    AEDataExtractionStep,
    DataExtractionStep,
    USERS,
    SERVICES,
    NUM_OF_USERS,
    NUM_OF_SERVICES,
    AUTOENCODERS,
)
from recommender.models.user import User
from recommender.models.service import Service
from tests.conftest import users_services_args


def test_data_extraction_step(mongo, generate_users_and_services, ae_pipeline_config):
    """
    Testing:
    -> configuration
    -> the number of generated users/services
    -> the class of generated users/services
    """
    ae_data_extraction_step = AEDataExtractionStep(ae_pipeline_config)
    # Check the correctness of the configuration inside data extraction step
    assert (
        ae_pipeline_config[DataExtractionStep.__name__]
        == ae_data_extraction_step.config
    )

    data, details = ae_data_extraction_step()

    # The expected number of users/services
    args = users_services_args()
    users_num = args["users_num"]
    services_num = args["common_services_num"] + args["unordered_services_num"]

    users = data[AUTOENCODERS][USERS]
    services = data[AUTOENCODERS][SERVICES]

    # The number of users/services from the data
    user_num_from_data = len(users)
    services_num_from_data = len(services)

    # The number of users/services from the details
    user_num_from_details = details[USERS][NUM_OF_USERS]
    services_num_from_details = details[SERVICES][NUM_OF_SERVICES]
    # Check the correctness of the number of users
    assert users_num == user_num_from_data == user_num_from_details
    # Check the correctness of the number of services
    assert services_num == services_num_from_data == services_num_from_details

    # Check if each returned users/services belong to the User/Service class
    for user in users:
        assert isinstance(user, User)

    for service in services:
        assert isinstance(service, Service)
