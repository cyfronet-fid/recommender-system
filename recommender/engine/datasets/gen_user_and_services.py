# pylint: disable=missing-module-docstring, too-many-arguments, too-many-locals

import random
from typing import Dict

from tqdm.auto import tqdm, trange

from recommender.models import Category, ScientificDomain
from tests.factories.marketplace import UserFactory, ServiceFactory
from tests.factories.marketplace.category import CategoryFactory
from tests.factories.marketplace.scientific_domain import ScientificDomainFactory


def gen_user_and_services(
    matching_services_numbers: Dict,
    cat_n: int = 10,
    sd_n: int = 10,
    users_n: int = 2000,
    user_cat_n: int = 5,
    user_sd_n: int = 5,
    services_n: int = 2000,
    service_cat_n: int = 5,
    service_sd_n: int = 5,
):
    """It generates given number of services and user, where the first user
    has relation to the generated services defined by `matching_services_numbers`.

    Args:
        matching_services_numbers: dict where key is the amount of common
            scientific domains and categories that generated service(s) should
            have in common with user. The value is the
            number of such services to generate.
        cat_n: total number of categories that will be assigned to users and services
        sd_n: total number of scientific domains that will
            be assigned to users and services
        users_n: number of users to generate
        user_cat_n: number of user's categories
        user_sd_n: number of user's scientific domains
        services_n: number of services to generate
        service_cat_n: number of service's categories
        service_sd_n: number of service's scientific domains
    """

    max_match = max(matching_services_numbers.keys())

    matching_services_number = sum(matching_services_numbers.values())
    if services_n <= matching_services_number:
        raise Exception("Too small services_n")

    not_matching_services_number = services_n - matching_services_number
    matching_services_numbers[0] = not_matching_services_number

    if cat_n <= user_cat_n or sd_n <= user_sd_n:
        raise Exception("Too small CAT_N or sd_n")

    if cat_n <= service_cat_n or sd_n <= service_sd_n:
        raise Exception("Too small CAT_N or sd_n")

    common = user_cat_n + user_sd_n
    if common < max_match:
        raise Exception(
            "Sum of categories and scientific domains number is less than max_match"
        )

    all_categories = CategoryFactory.create_batch(cat_n)
    user_categories = all_categories[:user_cat_n]
    not_user_categories = all_categories[user_cat_n:]

    all_scientific_domains = ScientificDomainFactory.create_batch(sd_n)
    user_scientific_domains = all_scientific_domains[:user_sd_n]
    not_user_scientific_domains = all_scientific_domains[user_sd_n:]

    user_common = user_categories + user_scientific_domains

    UserFactory(
        categories=user_categories,
        scientific_domains=user_scientific_domains,
        accessed_services=[],
        synthetic=True,
    )
    UserFactory.create_batch(users_n - 1, accessed_services=[], synthetic=True)

    total = len(matching_services_numbers.items())
    for match_n, amount in tqdm(matching_services_numbers.items(), total=total):
        for _ in trange(amount):
            common_sampled = random.sample(user_common, match_n)
            matching_c_sampled = list(
                filter(lambda obj: isinstance(obj, Category), common_sampled)
            )
            not_matching_c_sampled = random.sample(
                not_user_categories, service_cat_n - len(matching_c_sampled)
            )

            matching_sd_sampled = list(
                filter(lambda obj: isinstance(obj, ScientificDomain), common_sampled)
            )
            not_matching_sd_sampled = random.sample(
                not_user_scientific_domains, service_sd_n - len(matching_sd_sampled)
            )

            ServiceFactory(
                categories=matching_c_sampled + not_matching_c_sampled,
                scientific_domains=matching_sd_sampled + not_matching_sd_sampled,
            )
