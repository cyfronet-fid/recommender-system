# pylint: disable=too-many-arguments

"""This module contains functions useful for populating database with
 different types of data"""

import random
from tqdm.auto import trange

from tests.factories.marketplace import ServiceFactory, UserFactory


def populate_users_and_services(
    common_services_no,
    unordered_services_no,
    total_users,
    k_common_services_min,
    k_common_services_max,
    verbose=False,
):
    """Populate database with users and their ordered (or not) services"""

    _unordered_services = [
        ServiceFactory()
        for _ in trange(
            unordered_services_no,
            desc="Creating unordered services...",
            disable=(not verbose),
        )
    ]

    common_services = [
        ServiceFactory()
        for _ in trange(
            common_services_no,
            desc="Creating common services ...",
            disable=(not verbose),
        )
    ]

    for _ in trange(total_users, desc="Creating users...", disable=(not verbose)):
        k = random.randint(k_common_services_min, k_common_services_max)
        accessed_services = random.sample(common_services, k=k)
        user = UserFactory(accessed_services=accessed_services, synthetic=True)
        user.save()
