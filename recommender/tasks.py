# pylint: disable=missing-function-docstring, broad-except, fixme

"""This module contains celery tasks"""
from logger_config import get_logger

from recommender.engines.ncf.training.pipeline import NCFPipeline
from recommender.engines.rl.training.pipeline import RLPipeline
from recommender.extensions import celery
from recommender.pipeline_configs import (
    NCF_PIPELINE_CONFIG,
    RL_PIPELINE_CONFIG,
)
from recommender.services.deserializer import Deserializer
from recommender.services.drop_ml_models import drop_ml_models
from recommender.services.mp_dump import drop_mp_dump, load_mp_dump
from recommender.types import UserAction

logger = get_logger(__name__)


@celery.task
def update(data):
    try:
        drop_mp_dump()
        load_mp_dump(data)
        drop_ml_models()
        NCFPipeline(NCF_PIPELINE_CONFIG)()
        RLPipeline(RL_PIPELINE_CONFIG)()

    except Exception:
        logger.exception(
            "Exception has been raised. Training was unsuccessful. Aborting..."
        )
    else:
        logger.info("Training was successful")


@celery.task
def add_user_action(user_action_raw: dict):
    """
    Receive dict with user action, validate / parses it and saves it to DB
    """
    user_action = UserAction.parse_obj(user_action_raw)
    Deserializer.deserialize_user_action(user_action).save()
