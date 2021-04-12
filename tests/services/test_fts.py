# pylint: disable-all

from recommender.models import Service
from recommender.services.fts import retrieve_services
from tests.factories.marketplace import ServiceFactory
from tests.factories.marketplace.category import CategoryFactory
from tests.factories.marketplace.platform import PlatformFactory
from tests.factories.marketplace.provider import ProviderFactory
from tests.factories.marketplace.scientific_domain import ScientificDomainFactory
from tests.factories.marketplace.target_user import TargetUserFactory


def test_retrieve_services(mongo, mocker):
    ServiceFactory(
        id=0,
        name="Test name",
        description="Unique description",
        tagline="Extraordinary tagline",
        categories=[CategoryFactory(id=0), CategoryFactory(id=1)],
        countries=["WW"],  # World
        providers=[ProviderFactory(id=0), ProviderFactory(id=1)],
        platforms=[PlatformFactory(id=0), PlatformFactory(id=1)],
        scientific_domains=[
            ScientificDomainFactory(id=0),
            ScientificDomainFactory(id=1),
        ],
        target_users=[TargetUserFactory(id=0), TargetUserFactory(id=1)],
    )

    ServiceFactory(
        id=1,
        name="Nice name",
        description="Very unique description",
        tagline="Tested tagline",
        categories=[CategoryFactory(id=1)],
        countries=["PL"],  # Poland
        providers=[ProviderFactory(id=1)],
        platforms=[PlatformFactory(id=0), PlatformFactory(id=1)],
        scientific_domains=[ScientificDomainFactory(id=0)],
        target_users=[TargetUserFactory(id=2)],
    )

    # We have to mock search_text because it is not implemented in mongomock
    search_text_mock = mocker.patch(
        "mongoengine.queryset.queryset.QuerySet.search_text"
    )
    search_text_mock.return_value = Service.objects

    # Mocked search_text call
    assert [x.id for x in retrieve_services({"q": "EGI"})] == [0, 1]
    search_text_mock.assert_called_once_with("EGI")

    assert [x.id for x in retrieve_services({"categories": [0]})] == [0]
    assert [x.id for x in retrieve_services({"categories": [1]})] == [0, 1]
    assert [x.id for x in retrieve_services({"countries": ["PL", "WW", "EU"]})] == [
        0,
        1,
    ]
    assert [x.id for x in retrieve_services({"providers": [0]})] == [0]
    assert [x.id for x in retrieve_services({"providers": [1]})] == [0, 1]
    assert [x.id for x in retrieve_services({"providers": [0, 1]})] == [0, 1]
    assert [x.id for x in retrieve_services({"platforms": [0, 1]})] == [0, 1]
    assert [x.id for x in retrieve_services({"platforms": [0]})] == [0, 1]
    assert [x.id for x in retrieve_services({"platforms": [1]})] == [0, 1]
    assert [x.id for x in retrieve_services({"scientific_domains": [1]})] == [0]
    assert [x.id for x in retrieve_services({"scientific_domains": [0]})] == [0, 1]
    assert [x.id for x in retrieve_services({"target_users": [0]})] == [0]
    assert [x.id for x in retrieve_services({"target_users": [0, 1]})] == [0]
    assert [x.id for x in retrieve_services({"target_users": [1]})] == [0]
    assert [x.id for x in retrieve_services({"target_users": [2]})] == [1]
    assert [x.id for x in retrieve_services({"target_users": [3]})] == []
    assert [
        x.id for x in retrieve_services({"categories": [0], "target_users": [2]})
    ] == []
    assert [
        x.id for x in retrieve_services({"categories": [1], "target_users": [2]})
    ] == [1]
