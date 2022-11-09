# pylint: disable=missing-function-docstring, broad-except, fixme

"""This module contains celery tasks"""
import json

import stomp
from flask import current_app

from logger_config import get_logger
from recommender.engines.autoencoders.inference.embedding_component import (
    EmbeddingComponent,
)
from recommender.engines.autoencoders.training.pipeline import AEPipeline
from recommender.engines.ncf.training.pipeline import NCFPipeline
from recommender.engines.rl.training.pipeline import RLPipeline
from recommender.extensions import celery
from recommender.pipeline_configs import (
    AUTOENCODERS_PIPELINE_CONFIG,
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

        AEPipeline(AUTOENCODERS_PIPELINE_CONFIG)()
        EmbeddingComponent()()
        # TODO: parallel computing algebra here to make it faster
        # TODO: Commented unused pipelines to speed up the training
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


@celery.task
def send_recommendation_to_databus(context: dict, recommendation_response: dict):
    host = current_app.config["RS_DATABUS_HOST"]
    port = current_app.config["RS_DATABUS_PORT"]
    username = current_app.config["RS_DATABUS_USERNAME"]
    password = current_app.config["RS_DATABUS_PASSWORD"]
    publish_topic = current_app.config["RS_DATABUS_PUBLISH_TOPIC"]
    enable_ssl = current_app.config["RS_DATABUS_SSL"]

    conn = stomp.Connection([(host, port)])
    conn.connect(username, password, wait=True)

    conn.send(
        body=json.dumps(
            {
                "recommender_system": "cyfronet",
                "context": context,
                "response": recommendation_response,
            }
        ),
        destination=f"{publish_topic}",
        headers={"content-type": "application/json"},
        ssl=enable_ssl,
    )

    conn.disconnect()
