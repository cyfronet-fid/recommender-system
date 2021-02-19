# pylint: disable=expression-not-assigned

"""This module contains logic for seeding factories fakers.
 To make factories more realistic you can use seed_faker function
 to generate special json files from data existing in the database.
 These files will contain information used by factories' fakers
 to generate more realistic data"""
from mongoengine import connect

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
from settings import DevelopmentConfig

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


if __name__ == "__main__":
    connection = connect(host=DevelopmentConfig.MONGODB_HOST)

    print("Seeding fakers...")
    seed_faker()
    print("Fakers seeded successfully!\n")
