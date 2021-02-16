# pylint: disable-all

import time
from datetime import datetime

from faker import Factory as FakerFactory
import factory.random

from app.db.mongo_models import UserAction

from .marketplace_factories import UserFactory
from tests.factories.user_action.target_factory import TargetFactory
from tests.factories.user_action.source_factory import SourceFactory
from tests.factories.user_action.action_factory import ActionFactory

faker = FakerFactory.create()


class UserActionFactory(factory.mongoengine.MongoEngineFactory):
    class Meta:
        model = UserAction

    logged_user = True
    user = factory.SubFactory(UserFactory)
    unique_id = None
    timestamp = factory.LazyFunction(lambda: datetime.fromtimestamp(time.time()))
    source = factory.SubFactory(SourceFactory)
    target = factory.SubFactory(TargetFactory)
    action = factory.SubFactory(ActionFactory)

    class Params:
        recommendation_root = factory.Trait(
            source=factory.SubFactory(SourceFactory, recommendation_root=True)
        )
        regular_list_root = factory.Trait(
            source=factory.SubFactory(SourceFactory, regular_list_root=True)
        )

        not_logged = factory.Trait(
            logged_user=False,
            user=None,
            unique_id=factory.LazyFunction(lambda: faker.uuid4(cast_to=None).int >> 65),
        )
        order = factory.Trait(action=factory.SubFactory(ActionFactory, order=True))
