# pylint: disable=too-few-public-methods, missing-function-docstring

"""File containing configs for all the environments"""

import os

import torch


def get_device(env_variable):
    device_name = os.environ.get(env_variable, "cpu")
    if device_name == "cuda":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"

    return device_name


class Config:
    """Default config"""

    SWAGGER_UI_DOC_EXPANSION = "list"
    RESTPLUS_VALIDATE = True
    RESTPLUS_ERROR_404_HELP = False
    RESTPLUS_MASK_SWAGGER = False
    REDIS_HOST = f"redis://{os.environ.get('REDIS_HOST', '127.0.0.1:6379')}"
    TRAINING_DEVICE = get_device("TRAINING_DEVICE")
    RS_DATABUS_HOST = os.environ.get("RS_DATABUS_HOST", "127.0.0.1")
    RS_DATABUS_PORT = int(os.environ.get("RS_DATABUS_PORT", 61613))
    RS_DATABUS_USERNAME = os.environ.get("RS_DATABUS_USERNAME", "admin")
    RS_DATABUS_PASSWORD = os.environ.get("RS_DATABUS_PASSWORD", "admin")
    RS_DATABUS_SSL = not os.environ.get("RS_DATABUS_SSL") in ["0", "no"]
    RS_DATABUS_PUBLISH_TOPIC = os.environ.get(
        "RS_DATABUS_PUBLISH_TOPIC", "/topic/recommendations"
    )
    RS_DATABUS_SUBSCRIPTION_TOPIC = os.environ.get(
        "RS_DATABUS_SUBSCRIPTION_TOPIC", "/topic/user_actions"
    )


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
    MONGODB_HOST = f"mongodb://{os.environ.get('MONGODB_HOST', '127.0.0.1:27017')}/test"
    RS_DATABUS_HOST = os.environ.get("TEST_RS_DATABUS_HOST", "127.0.0.1")
    RS_DATABUS_PORT = int(os.environ.get("TEST_RS_DATABUS_PORT", 61613))
    RS_DATABUS_USERNAME = os.environ.get("TEST_RS_DATABUS_USERNAME", "admin")
    RS_DATABUS_PASSWORD = os.environ.get("TEST_RS_DATABUS_PASSWORD", "admin")
    RS_DATABUS_PUBLISH_TOPIC = os.environ.get(
        "TEST_RS_DATABUS_PUBLISH_TOPIC", "/topic/recommendations_test"
    )
    RS_DATABUS_SUBSCRIPTION_TOPIC = os.environ.get(
        "TEST_RS_DATABUS_SUBSCRIPTION_TOPIC", "topic/user_actions_test"
    )


config_by_name = dict(
    development=DevelopmentConfig, testing=TestingConfig, production=ProductionConfig
)
