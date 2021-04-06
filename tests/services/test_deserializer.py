# pylint: disable-all

from copy import deepcopy
from recommender.models import UserAction
from recommender.models import Recommendation
from recommender.services.deserializer import Deserializer
from tests.factories.marketplace import UserFactory, ServiceFactory
from tests.factories.marketplace.category import CategoryFactory
from tests.factories.marketplace.platform import PlatformFactory
from tests.factories.marketplace.provider import ProviderFactory
from tests.factories.marketplace.scientific_domain import ScientificDomainFactory
from tests.factories.marketplace.target_user import TargetUserFactory


def test_recommendation_deserialization(mongo, recommendation_json_dict):
    UserFactory(id=1)
    [ServiceFactory(id=i) for i in range(3)]
    CategoryFactory(id=1)
    ProviderFactory(id=1)
    PlatformFactory(id=1)
    ScientificDomainFactory(id=1)
    TargetUserFactory(id=1)
    Deserializer.deserialize_recommendation(recommendation_json_dict).save()
    r = Recommendation.objects.first()

    assert r.user.id == recommendation_json_dict.get("user_id")
    assert str(r.unique_id) == recommendation_json_dict.get("unique_id")
    assert str(r.timestamp.isoformat(timespec='milliseconds'))+'Z' == recommendation_json_dict.get("timestamp")
    assert str(r.visit_id) == recommendation_json_dict.get("visit_id")
    assert [s.id for s in r.services] == recommendation_json_dict.get("services")

    search_data_json_dict = recommendation_json_dict.get("search_data")
    assert r.search_data.q == search_data_json_dict.get("q")
    assert r.search_data.category.id == search_data_json_dict.get("category_id")
    assert r.search_data.geographical_availabilities == search_data_json_dict.get(
        "geographical_availabilities"
    )
    assert r.search_data.order_type == search_data_json_dict.get("order_type")
    assert [p.id for p in r.search_data.providers] == search_data_json_dict.get("providers")
    assert [rp.id for rp in r.search_data.related_platforms] == search_data_json_dict.get("related_platforms")
    assert [sd.id for sd in r.search_data.scientific_domains] == search_data_json_dict.get("scientific_domains")
    assert r.search_data.sort == search_data_json_dict.get("sort")
    assert [tu.id for tu in r.search_data.target_users] == search_data_json_dict.get("target_users")


def test_user_action_deserialization(mongo, user_action_json_dict):
    UserFactory(id=1)
    ServiceFactory(id=1)
    Deserializer.deserialize_user_action(user_action_json_dict).save()
    ua = UserAction.objects.first()

    assert ua.user.id == user_action_json_dict.get("user_id")
    assert str(ua.unique_id) == user_action_json_dict.get("unique_id")
    assert str(ua.timestamp.isoformat(timespec='milliseconds')) + 'Z' == user_action_json_dict.get("timestamp")
    assert str(ua.source.visit_id) == user_action_json_dict.get("source").get("visit_id")
    assert ua.source.page_id == user_action_json_dict.get("source").get("page_id")
    assert ua.source.root.type == user_action_json_dict.get("source").get("root").get("type")
    assert ua.source.root.panel_id == user_action_json_dict.get("source").get("root").get("panel_id")
    assert ua.source.root.service.id == user_action_json_dict.get("source").get("root").get("service_id")
    assert str(ua.target.visit_id) == user_action_json_dict.get("target").get("visit_id")
    assert ua.target.page_id == user_action_json_dict.get("target").get("page_id")
    assert ua.action.type == user_action_json_dict.get("action").get("type")
    assert ua.action.text == user_action_json_dict.get("action").get("text")
    assert ua.action.order == user_action_json_dict.get("action").get("order")

    json_dict_without_root = deepcopy(user_action_json_dict)
    json_dict_without_root.get("source").pop("root")
    Deserializer.deserialize_user_action(json_dict_without_root).save()
    ua = UserAction.objects[1]

    assert ua.user.id == user_action_json_dict.get("user_id")
    assert str(ua.unique_id) == user_action_json_dict.get("unique_id")
    assert str(ua.timestamp.isoformat(timespec='milliseconds')) + 'Z' == user_action_json_dict.get("timestamp")
    assert str(ua.source.visit_id) == user_action_json_dict.get("source").get("visit_id")
    assert ua.source.page_id == user_action_json_dict.get("source").get("page_id")
    assert str(ua.target.visit_id) == user_action_json_dict.get("target").get("visit_id")
    assert ua.target.page_id == user_action_json_dict.get("target").get("page_id")
    assert ua.action.type == user_action_json_dict.get("action").get("type")
    assert ua.action.text == user_action_json_dict.get("action").get("text")
    assert ua.action.order == user_action_json_dict.get("action").get("order")
