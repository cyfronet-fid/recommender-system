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
    RL_PIPELINE_CONFIG,
)
from recommender.commands.db import drop_ml_models


def _ae():
    """Run AutoEncoders pipeline"""
    AEPipeline(AUTOENCODERS_PIPELINE_CONFIG)()


def _embedding():
    """Embed User and Services and save dense_tensors to cache"""
    EmbeddingComponent()(verbose=True)


def _ncf():
    """Run NCF pipeline"""
    NCFPipeline(NCF_PIPELINE_CONFIG)()


def _rl():
    """Run RL (TD3) pipeline"""
    RLPipeline(RL_PIPELINE_CONFIG)()


def _all():
    """Delete all old ML models and run all training pipelines"""
    drop_ml_models()
    AEPipeline(AUTOENCODERS_PIPELINE_CONFIG)()
    EmbeddingComponent()(verbose=True)
    NCFPipeline(NCF_PIPELINE_CONFIG)()
    RLPipeline(RL_PIPELINE_CONFIG)()


def train_command(task):
    globals()[f"_{task}"]()
