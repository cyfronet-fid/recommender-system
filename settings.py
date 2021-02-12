# pylint: disable=too-few-public-methods

"""File containing configs for all the environments"""

import os


class Config:
    """Default config"""

    SWAGGER_UI_DOC_EXPANSION = "list"
    RESTPLUS_VALIDATE = True
    RESTPLUS_ERROR_404_HELP = False
    RESTPLUS_MASK_SWAGGER = False
    REDIS_HOST = f"redis://{os.environ.get('REDIS_HOST', '127.0.0.1:6379')}"


class ProductionConfig(Config):
    """Production config"""

    RESTPLUS_ERROR_404_HELP = True
    MONGODB_HOST = (
        f"mongodb://{os.environ.get('MONGODB_HOST', '127.0.0.1:27017')}/recommender"
    )


class DevelopmentConfig(Config):
    """Development config"""

    DEBUG = True
    MONGODB_HOST = (
        f"mongodb://{os.environ.get('MONGODB_HOST', '127.0.0.1:27017')}/recommender_dev"
    )


class TestingConfig(Config):
    """Testing config"""

    TESTING = True
    MONGODB_HOST = "mongomock://localhost/test"


config_by_name = dict(
    development=DevelopmentConfig, testing=TestingConfig, production=ProductionConfig
)
