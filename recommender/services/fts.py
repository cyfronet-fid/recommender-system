# pylint: disable=invalid-name, no-member

"""Mongo FTS Operations module"""

from recommender.models import Service

AVAILABLE_FOR_RECOMMENDATION = ("published", "unverified")


def retrieve_services(search_data, accessed_services_ids=None):
    """Applies search info from MP and filters MongoDB by them"""
    categories = search_data.get("categories")
    countries = search_data.get("geographical_availabilities")
    provider_ids = search_data.get("providers")
    search_phrase = search_data.get("q")
    platform_ids = search_data.get("related_platforms")
    scientific_domain_ids = search_data.get("scientific_domains")
    target_user_ids = search_data.get("target_users")
    q = Service.objects
    q = q(categories__in=categories) if categories is not None else q
    q = q(countries__in=countries) if countries else q
    q = q(providers__in=provider_ids) if provider_ids else q
    q = q(platforms__in=platform_ids) if platform_ids else q
    q = q(scientific_domains__in=scientific_domain_ids) if scientific_domain_ids else q
    q = q(target_users__in=target_user_ids) if target_user_ids else q
    q = q.search_text(search_phrase) if search_phrase else q
    q = q(id__nin=accessed_services_ids) if accessed_services_ids else q
    q = q(status__in=AVAILABLE_FOR_RECOMMENDATION)

    print(f"services available for recommendations: {len(q)}")
    return q
