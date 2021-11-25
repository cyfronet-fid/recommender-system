# pylint: disable=missing-function-docstring

"""Flask CLI train commands"""

from recommender.engines.autoencoders.inference.embedding_component import (
    EmbeddingComponent,
)
from recommender.engines.autoencoders.training.pipeline import AEPipeline
from recommender.engines.ncf.training.pipeline import NCFPipeline
from recommender.engines.rl.training.pipeline import RLPipeline
from recommender.pipeline_configs import (
    AUTOENCODERS_PIPELINE_CONFIG,
    NCF_PIPELINE_CONFIG,
    RL_PIPELINE_CONFIG_V1,
    RL_PIPELINE_CONFIG_V2,
)


def _ae():
    """Run AutoEncoders pipeline"""
    AEPipeline(AUTOENCODERS_PIPELINE_CONFIG)()


def _embedding():
    """Embed User and Services and save dense_tensors to cache"""
    EmbeddingComponent()()


def _ncf():
    """Run NCF pipeline"""
    NCFPipeline(NCF_PIPELINE_CONFIG)()


def _rl_v1():
    """Run RL (TD3) pipeline in version 1"""
    RLPipeline(RL_PIPELINE_CONFIG_V1)()


def _rl_v2():
    """Run RL (TD3) pipeline in version 1"""
    RLPipeline(RL_PIPELINE_CONFIG_V2)()


def _all():
    """Run all training pipelines"""
    AEPipeline(AUTOENCODERS_PIPELINE_CONFIG)()
    EmbeddingComponent()()
    NCFPipeline(NCF_PIPELINE_CONFIG)()
    RLPipeline(RL_PIPELINE_CONFIG_V1)()
    RLPipeline(RL_PIPELINE_CONFIG_V2)()


def train_command(task):
    globals()[f"_{task}"]()
