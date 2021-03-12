"""This module contains functions useful for populating database with
 different types of data"""

import random
from tqdm.auto import trange

from tests.factories.marketplace import ServiceFactory, UserFactory


def populate_users_and_services(
    common_services_number,
    no_one_services_number,
    users_number,
    k_common_services_min,
    k_common_services_max,
):
    """Populate database with users and their accessed (or not) services"""

    _no_one_services = [
        ServiceFactory()
        for _ in trange(no_one_services_number, desc="No one services creating...")
    ]

    common_services = [
        ServiceFactory()
        for _ in trange(common_services_number, desc="Common services creating...")
    ]

    for _ in trange(users_number, desc="Users creating..."):
        k = random.randint(k_common_services_min, k_common_services_max)
        accessed_services = random.sample(common_services, k=k)
        user = UserFactory(accessed_services=accessed_services)
        user.save()
