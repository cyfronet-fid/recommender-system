"""Module for the flask CLI commands"""

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
from recommender.services.mp_dump import drop_mp_dump

from tests.factories.marketplace.faker_seeds.utils.dumpers import (
    dump_names_descs,
    dump_names,
    dump_taglines,
)
from tests.factories.populate_database import populate_users_and_services


def seed_faker():
    """
    Call this function to generate json files used for seeding fakers in factories.
    It uses current database data for it. To make factories more
    realistic you can use seed_faker function to generate special
    json files from data existing in the database.
    These files will contain information used by factories'
    fakers to generate more realistic data
    """
    names_descs_classes = [
        Service,
        AccessMode,
        AccessType,
        LifeCycleStatus,
        TargetUser,
        Trl,
    ]
    class_names = [Category, Platform, Provider, ScientificDomain]

    for clazz in names_descs_classes:
        dump_names_descs(clazz)
    for clazz in class_names:
        dump_names(clazz)

    dump_taglines()


def execute_training(task):
    """Runs given machine learning task"""
    if task == "ae":
        AEPipeline(AUTOENCODERS_PIPELINE_CONFIG)()
    elif task == "embedding":
        EmbeddingComponent()()
    elif task == "ncf":
        NCFPipeline(NCF_PIPELINE_CONFIG)()
    elif task == "rl":
        RLPipeline(RL_PIPELINE_CONFIG_V1)()
    elif task == "all":
        AEPipeline(AUTOENCODERS_PIPELINE_CONFIG)()
        EmbeddingComponent()()
        NCFPipeline(NCF_PIPELINE_CONFIG)()
        RLPipeline(RL_PIPELINE_CONFIG_V1)()
        RLPipeline(RL_PIPELINE_CONFIG_V2)()


def seed_db():
    """Populates databse with a small amount of data
    for testing and development purposes"""
    populate_users_and_services(
        common_services_no=30,
        unordered_services_no=70,
        total_users=100,
        k_common_services_max=10,
        k_common_services_min=1,
        verbose=True,
    )


def drop_mp_dump_task():
    """Drops the documents sent by the MP Database dump"""
    drop_mp_dump()
