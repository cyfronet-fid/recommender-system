# pylint: disable-all

from factory import Trait, SubFactory
from factory.mongoengine import MongoEngineFactory

from recommender.models import Root
from tests.factories.marketplace import ServiceFactory


class RootFactory(MongoEngineFactory):
    class Meta:
        model = Root

    type = "recommendation_panel"
    panel_id = "services_catalogue_list"
    service = SubFactory(ServiceFactory)

    class Params:
        recommendation_panel = Trait(
            type="recommendation_panel", panel_id="services_catalogue_list"
        )
        regular_services_list = Trait(type="regular_services_list", panel_id=None)
