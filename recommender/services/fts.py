# pylint: disable=invalid-name, no-member

"""Mongo FTS Operations module"""

from typing import Iterable, Optional, List, Tuple
from mongoengine import QuerySet
from recommender.models import Service, SearchData

AVAILABLE_FOR_RECOMMENDATION = ("published", "unverified")


def filter_services(search_data: SearchData) -> QuerySet:
    """
    Filter services based on provided search data.

    Args:
        search_data: Search Data object.

    Returns:
        q: Query set
    """

    categories = search_data.categories
    countries = search_data.geographical_availabilities
    providers = search_data.providers
    platforms = search_data.related_platforms
    scientific_domains = search_data.scientific_domains
    target_users = search_data.target_users
    search_phrase = search_data.q

    q = Service.objects
    q = q(categories__in=categories) if len(categories) > 0 else q
    q = q(countries__in=countries) if len(countries) > 0 else q
    q = q(providers__in=providers) if len(providers) > 0 else q
    q = q(platforms__in=platforms) if len(platforms) > 0 else q
    q = (
        q(scientific_domains__in=scientific_domains)
        if len(scientific_domains) > 0
        else q
    )
    q = q(target_users__in=target_users) if len(target_users) > 0 else q
    q = q.search_text(search_phrase) if search_phrase not in (None, "") else q
    return q


def retrieve_services_for_recommendation(
    elastic_services: Tuple[int], accessed_services: Optional[Iterable] = None
) -> QuerySet:
    """
    Selecting candidates for recommendation

    Args:
        elastic_services: Marketplace's context
        accessed_services: Services that user accessed
    """
    q = list(elastic_services)
    q = filter_available_and_recommendable_services(q)
    q = q(id__nin=[s.id for s in accessed_services]) if accessed_services else q
    return q


def filter_available_and_recommendable_services(q: List[int]) -> QuerySet:
    """
    - Check if services exist in the RS database.
    - Check if the status of a service allows the service to be recommended
    """
    q = Service.objects(id__in=q)
    q = q(status__in=AVAILABLE_FOR_RECOMMENDATION)
    return q


def retrieve_services_for_synthetic_sarses(
    search_data: SearchData, accessed_services: Optional[Iterable] = None
):
    """Applies search info from MP and filters MongoDB by them"""

    q = filter_services(search_data)
    q = q(id__nin=[s.id for s in accessed_services]) if accessed_services else q
    q = q(status__in=AVAILABLE_FOR_RECOMMENDATION)
    return q


def retrieve_forbidden_services():
    """Returns services that should not be recommended"""
    q = Service.objects(status__nin=AVAILABLE_FOR_RECOMMENDATION)
    return q
