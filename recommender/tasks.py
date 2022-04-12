# pylint: disable=missing-function-docstring, broad-except, fixme

"""This module contains celery tasks"""
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
from recommender.services.mp_dump import drop_mp_dump, load_mp_dump
from recommender.services.drop_ml_models import drop_ml_models
from logger_config import get_logger

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
