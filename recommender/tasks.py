# pylint: disable=missing-function-docstring, fixme

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
    RL_PIPELINE_CONFIG_V2,
)
from recommender.services.mp_dump import drop_mp_dump, load_mp_dump


@celery.task
def update(data):
    drop_mp_dump()
    load_mp_dump(data)
    AEPipeline(AUTOENCODERS_PIPELINE_CONFIG)()
    EmbeddingComponent()()
    # TODO: parallel computing algebra here to make it faster
    NCFPipeline(NCF_PIPELINE_CONFIG)()
    RLPipeline(RL_PIPELINE_CONFIG_V1)()
    RLPipeline(RL_PIPELINE_CONFIG_V2)()
