# pylint: disable=too-many-arguments, line-too-long

"""This module contains functions useful for populating database with
 different types of data"""

import random
from tqdm.auto import trange

from tests.factories.marketplace import ServiceFactory, UserFactory
from recommender.errors import RangeOfCommonServicesError


def populate_users_and_services(
    common_services_num: int,
    unordered_services_num: int,
    users_num: int,
    k_common_services_min: int,
    k_common_services_max: int,
    verbose: bool = False,
    valid: bool = True,
):
    """
    Populate database with users and their ordered (or not) services

    Args:
        common_services_num:    how many common services?
        unordered_services_num: how many unordered services?
        users_num:              how many users?
        k_common_services_min:  the minimum common services for a user
        k_common_services_max:  the maximum common services for a user
        verbose:                be verbose?
        valid:                  should created users and services be valid (have categories and scientific domains)?
    """
    if k_common_services_min > k_common_services_max:
        raise RangeOfCommonServicesError()

    _unordered_services = [
        ServiceFactory()
        for _ in trange(
            unordered_services_num,
            desc="Creating unordered services...",
            disable=(not verbose),
        )
    ]

    common_services = [
        ServiceFactory()
        for _ in trange(
            common_services_num,
            desc="Creating common services ...",
            disable=(not verbose),
        )
    ]

    if not valid:
        # Create invalid services which do not have any categories and scientific_domains
        for service in _unordered_services:
            service.categories = []
            service.scientific_domains = []
        for service in common_services:
            service.categories = []
            service.scientific_domains = []

    for _ in trange(users_num, desc="Users creating...", disable=(not verbose)):
        k = random.randint(k_common_services_min, k_common_services_max)
        accessed_services = random.sample(common_services, k=k)
        user = UserFactory(accessed_services=accessed_services, synthetic=True)
        if not valid:
            # Create invalid users
            user.categories = []
            user.scientific_domains = []
        user.save()
