# pylint: disable=missing-function-docstring

"""Flask CLI seed commands"""
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
from tests.factories.marketplace.faker_seeds.utils.dumpers import (
    dump_names_descs,
    dump_names,
    dump_taglines,
)
from tests.factories.populate_database import populate_users_and_services


def _seed():
    """Populates database with a small amount of data
    for testing and development purposes"""
    populate_users_and_services(
        common_services_num=30,
        unordered_services_num=70,
        users_num=100,
        k_common_services_min=1,
        k_common_services_max=10,
        verbose=True,
        valid=True,
    )


def _seed_faker():
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


def seed_command(task):
    globals()[f"_{task}"]()
