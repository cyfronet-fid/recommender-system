# pylint: disable=missing-function-docstring

"""Flask CLI train commands"""

from recommender.engines.ncf.training.pipeline import NCFPipeline
from recommender.engines.rl.training.pipeline import RLPipeline
from recommender.pipeline_configs import (
    NCF_PIPELINE_CONFIG,
    RL_PIPELINE_CONFIG,
)
from recommender.commands.db import drop_ml_models


def _ncf():
    """Run NCF pipeline"""
    NCFPipeline(NCF_PIPELINE_CONFIG)()


def _rl():
    """Run RL (TD3) pipeline"""
    RLPipeline(RL_PIPELINE_CONFIG)()


def _all():
    """Delete all old ML models and run all training pipelines"""
    drop_ml_models()
    NCFPipeline(NCF_PIPELINE_CONFIG)()
    RLPipeline(RL_PIPELINE_CONFIG)()


def train_command(task):
    globals()[f"_{task}"]()
