# pylint: disable=expression-not-assigned, fixme, missing-function-docstring

"""This module contains logic for seeding factories fakers.
 To make factories more realistic you can use seed_faker function
 to generate special json files from data existing in the database.
 These files will contain information used by factories' fakers
 to generate more realistic data"""
from recommender.engines.autoencoders.inference.embedding_component import (
    EmbeddingComponent,
)
from recommender.engines.autoencoders.training.pipeline import AEPipeline
from recommender.engines.ncf.training.pipeline import NCFPipeline
from recommender.engines.rl.training.pipeline import RLPipeline
from recommender.models import (
    Service,
    AccessMode,
    AccessType,
    LifeCycleStatus,
    TargetUser,
    Trl,
    Category,
    Platform,
    Provider,
    ScientificDomain,
)
from recommender.pipeline_configs import (
    RL_PIPELINE_CONFIG_V1,
    AUTOENCODERS_PIPELINE_CONFIG,
    NCF_PIPELINE_CONFIG,
    RL_PIPELINE_CONFIG_V2,
)

from tests.factories.marketplace.faker_seeds.utils.dumpers import (
    dump_names_descs,
    dump_names,
    dump_taglines,
)

NAMES_DESCS_CLASSES = [
    Service,
    AccessMode,
    AccessType,
    LifeCycleStatus,
    TargetUser,
    Trl,
]
NAMES_CLASSES = [Category, Platform, Provider, ScientificDomain]


def seed_faker():
    """Call this function to generate json files used for seeding fakers in factories.
    It uses current database data for it."""
    [dump_names_descs(clazz) for clazz in NAMES_DESCS_CLASSES]
    [dump_names(clazz) for clazz in NAMES_CLASSES]

    dump_taglines()


def execute_pipelines():
    AEPipeline(AUTOENCODERS_PIPELINE_CONFIG)()
    EmbeddingComponent()()
    NCFPipeline(NCF_PIPELINE_CONFIG)()
    RLPipeline(RL_PIPELINE_CONFIG_V1)()
    RLPipeline(RL_PIPELINE_CONFIG_V2)()


def train_rl_pipeline():
    RLPipeline(RL_PIPELINE_CONFIG_V1)()
