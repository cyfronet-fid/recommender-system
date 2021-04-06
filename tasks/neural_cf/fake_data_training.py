# pylint: disable=redefined-outer-name, redefined-builtin, missing-function-docstring, no-member


"""Use this task to train Neural Colaborative Filtering model on fake data
 (automaticaly generated in the testing database). Trained model will be
 save into development database.
 """

from mongoengine import connect, disconnect

from recommender.engine.pre_agent.models import NEURAL_CF
from recommender.engine.pre_agent.models import load_last_module
from recommender.engine.pre_agent.training.common import pre_agent_training
from recommender.engine.pre_agent.datasets.all_datasets import create_datasets
from recommender.engine.pre_agent.models import save_module
from recommender.engine.pre_agent.preprocessing import (
    precalc_users_and_service_tensors,
    load_last_transformer,
    USERS,
    SERVICES,
    save_transformer,
)
from settings import TestingConfig, DevelopmentConfig
from tests.factories.populate_database import populate_users_and_services

if __name__ == "__main__":
    connect(host=TestingConfig.MONGODB_HOST)

    COMMON_SERVICES_NUMBER = 10
    NO_ONE_SERVICES_NUMBER = int(COMMON_SERVICES_NUMBER / 10)
    USERS_NUMBER = 10
    K_COMMON_SERVICES_MIN = 3
    K_COMMON_SERVICES_MAX = 7

    populate_users_and_services(
        common_services_number=COMMON_SERVICES_NUMBER,
        no_one_services_number=NO_ONE_SERVICES_NUMBER,
        users_number=USERS_NUMBER,
        k_common_services_min=K_COMMON_SERVICES_MIN,
        k_common_services_max=K_COMMON_SERVICES_MAX,
    )

    precalc_users_and_service_tensors()
    create_datasets()
    pre_agent_training()

    model = load_last_module(NEURAL_CF)
    user_transformer = load_last_transformer(USERS)
    service_transformer = load_last_transformer(SERVICES)

    disconnect()
    connect(host=DevelopmentConfig.MONGODB_HOST)

    save_module(model, NEURAL_CF)
    save_transformer(user_transformer, USERS)
    save_transformer(service_transformer, SERVICES)
