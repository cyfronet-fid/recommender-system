# pylint: disable-all

import time
from datetime import datetime
from factory import SubFactory, Trait, LazyFunction
from factory.mongoengine import MongoEngineFactory
from faker import Factory as FakerFactory

from recommender.models import UserAction
from tests.factories.marketplace import UserFactory
from .target import TargetFactory
from .source import SourceFactory
from .action import ActionFactory

faker = FakerFactory.create()


class UserActionFactory(MongoEngineFactory):
    class Meta:
        model = UserAction

    user = SubFactory(UserFactory)
    unique_id = LazyFunction(lambda: faker.uuid4())
    timestamp = LazyFunction(lambda: datetime.fromtimestamp(time.time()))
    source = SubFactory(SourceFactory)
    target = SubFactory(TargetFactory)
    action = SubFactory(ActionFactory)

    class Params:
        recommendation_root = Trait(
            source=SubFactory(SourceFactory, recommendation_root=True)
        )
        regular_list_root = Trait(
            source=SubFactory(SourceFactory, regular_list_root=True)
        )

        not_logged = Trait(
            user=None
        )
        order = Trait(action=SubFactory(ActionFactory, order=True))
