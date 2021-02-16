# pylint: disable-all


import factory
from app.db.mongo_models import Root
from tests.factories.marketplace_factories import ServiceFactory


class RootFactory(factory.mongoengine.MongoEngineFactory):
    class Meta:
        model = Root

    type = "recommendation_panel"
    panel_id = "services_catalogue_list"
    service = factory.SubFactory(ServiceFactory)

    class Params:
        recommendation_panel = factory.Trait(
            type="recommendation_panel", panel_id="services_catalogue_list"
        )
        regular_services_list = factory.Trait(
            type="regular_services_list", panel_id=None
        )
