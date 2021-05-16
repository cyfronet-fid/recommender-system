# pylint: disable-all

from recommender.models import Service, SearchData
from recommender.services.fts import (
    retrieve_services_for_recommendation,
    retrieve_forbidden_services,
    filter_services,
)
from tests.factories.marketplace import ServiceFactory
from tests.factories.marketplace.category import CategoryFactory
from tests.factories.marketplace.platform import PlatformFactory
from tests.factories.marketplace.provider import ProviderFactory
from tests.factories.marketplace.scientific_domain import ScientificDomainFactory
from tests.factories.marketplace.target_user import TargetUserFactory


def test_filter_services(mongo, mocker):
    categories = CategoryFactory.create_batch(2)
    providers = ProviderFactory.create_batch(2)
    platforms = PlatformFactory.create_batch(2)
    scientific_domains = ScientificDomainFactory.create_batch(2)
    target_users = TargetUserFactory.create_batch(3)

    s0 = ServiceFactory(
        categories=categories,
        countries=["WW"],  # World
        providers=providers,
        resource_organisation=providers[0],
        platforms=platforms,
        scientific_domains=scientific_domains,
        target_users=target_users[:2],
    )

    s1 = ServiceFactory(
        categories=[categories[1]],
        countries=["PL"],  # Poland
        providers=[providers[1]],
        resource_organisation=providers[1],
        platforms=platforms,
        scientific_domains=[scientific_domains[0]],
        target_users=[target_users[2]],
    )

    # We have to mock search_text because it is not implemented in mongomock
    search_text_mock = mocker.patch(
        "mongoengine.queryset.queryset.QuerySet.search_text"
    )
    search_text_mock.return_value = Service.objects

    # Mocked search_text call
    assert list(filter_services(SearchData(q="EGI"))) == [s0, s1]
    search_text_mock.assert_called_once_with("EGI")

    assert list(filter_services(SearchData())) == [s0, s1]
    assert (
        list(filter_services(SearchData(categories=CategoryFactory.create_batch(2))))
        == []
    )

    assert list(filter_services(SearchData(categories=[categories[0]]))) == [s0]
    assert list(filter_services(SearchData(categories=[categories[1]]))) == [s0, s1]
    assert list(
        filter_services(SearchData(geographical_availabilities=["WW", "PL", "EU"]))
    ) == [s0, s1]
    assert list(filter_services(SearchData(providers=[providers[0]]))) == [s0]
    assert list(filter_services(SearchData(providers=[providers[1]]))) == [s0, s1]
    assert list(filter_services(SearchData(providers=providers))) == [s0, s1]
    assert list(filter_services(SearchData(related_platforms=platforms))) == [s0, s1]
    assert list(filter_services(SearchData(related_platforms=[platforms[0]]))) == [
        s0,
        s1,
    ]
    assert list(filter_services(SearchData(related_platforms=[platforms[1]]))) == [
        s0,
        s1,
    ]
    assert list(
        filter_services(SearchData(scientific_domains=[scientific_domains[1]]))
    ) == [s0]
    assert list(
        filter_services(SearchData(scientific_domains=[scientific_domains[0]]))
    ) == [s0, s1]
    assert list(filter_services(SearchData(target_users=[target_users[0]]))) == [s0]
    assert list(
        filter_services(SearchData(target_users=[target_users[0], target_users[1]]))
    ) == [s0]
    assert list(filter_services(SearchData(target_users=[target_users[1]]))) == [s0]
    assert list(filter_services(SearchData(target_users=[target_users[2]]))) == [s1]
    assert list(filter_services(SearchData(target_users=target_users))) == [s0, s1]
    assert (
        list(
            filter_services(
                SearchData(categories=[categories[0]], target_users=[target_users[2]])
            )
        )
        == []
    )
    assert list(
        filter_services(
            SearchData(categories=[categories[1]], target_users=[target_users[2]])
        )
    ) == [s1]


def test_retrieve_forbidden_services(mongo):
    services = [
        ServiceFactory(status="published"),
        ServiceFactory(status="unverified"),
        ServiceFactory(status="errored"),
        ServiceFactory(status="deleted"),
        ServiceFactory(status="draft"),
        ServiceFactory(status="unverified"),
    ]

    assert list(retrieve_forbidden_services()) == services[2:5]


def test_retrieve_services_for_recommendations(mongo):
    services = [
        ServiceFactory(status="published"),
        ServiceFactory(status="published"),
        ServiceFactory(status="unverified"),
        ServiceFactory(status="unverified"),
        ServiceFactory(status="deleted"),
        ServiceFactory(status="draft"),
    ]

    assert list(retrieve_services_for_recommendation(SearchData())) == services[:4]
    assert list(
        retrieve_services_for_recommendation(
            SearchData(), accessed_services=[services[0], services[2]]
        )
    ) == [services[1], services[3]]
