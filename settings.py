# pylint: disable=too-few-public-methods

"""File containing configs for all the environments"""

import os


class Config:
    """Default config"""

    SWAGGER_UI_DOC_EXPANSION = "list"
    RESTPLUS_VALIDATE = True
    RESTPLUS_ERROR_404_HELP = False
    RESTPLUS_MASK_SWAGGER = False
    CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379")
    CELERY_RESULT_BACKEND = os.environ.get(
        "CELERY_RESULT_BACKEND", "redis://localhost:6379"
    )


class ProductionConfig(Config):
    """Production config"""

    MONGODB_HOST = os.environ.get(
        "MONGODB_HOST", "mongodb://localhost:27017/recommender_prod"
    )


class DevelopmentConfig(Config):
    """Development config"""

    DEBUG = True
    MONGODB_HOST = os.environ.get(
        "MONGODB_HOST", "mongodb://localhost:27017/recommender_dev"
    )


class TestingConfig(Config):
    """Testing config"""

    TESTING = True
    MONGODB_HOST = "mongomock://localhost/test"


config_by_name = dict(
    development=DevelopmentConfig, testing=TestingConfig, production=ProductionConfig
)
