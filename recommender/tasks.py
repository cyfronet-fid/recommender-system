# pylint: disable=missing-function-docstring, broad-except, fixme

"""This module contain celery tasks"""
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
    RL_PIPELINE_CONFIG_V1,
)
from recommender.services.mp_dump import drop_mp_dump, load_mp_dump
from logger_config import get_logger

logger = get_logger(__name__)


@celery.task
def update(data):
    try:
        drop_mp_dump()
        load_mp_dump(data)
        AEPipeline(AUTOENCODERS_PIPELINE_CONFIG)()
        EmbeddingComponent()()

        # TODO: parallel computing algebra here to make it faster
        # TODO: Commented unused pipelines to speed up the training
        NCFPipeline(NCF_PIPELINE_CONFIG)()
        # RLPipeline(RL_PIPELINE_CONFIG_V2)()

        RLPipeline(RL_PIPELINE_CONFIG_V1)()

    except Exception:
        logger.exception(
            "Exception has been raised. Training was unsuccessful. Aborting..."
        )
    else:
        logger.info("Training was successful")
