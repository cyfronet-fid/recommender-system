# pylint: disable=redefined-outer-name, redefined-builtin, missing-function-docstring, no-member

"""Use this task to preprocess data in the development database before training.
 Make sure that data are present in the development database. If data has been
 loaded into database using database_dump endpoint then it is already preprocessed.
 """

from mongoengine import connect

from recommender.engine.pre_agent.datasets import create_datasets
from recommender.engine.pre_agent.preprocessing import precalc_users_and_service_tensors
from settings import DevelopmentConfig


if __name__ == "__main__":
    connect(host=DevelopmentConfig.MONGODB_HOST)
    precalc_users_and_service_tensors()
    create_datasets()
