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

    logged_user = True
    user = SubFactory(UserFactory)
    unique_id = None
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
            logged_user=False,
            user=None,
            unique_id=LazyFunction(lambda: faker.uuid4(cast_to=None).int >> 65),
        )
        order = Trait(action=SubFactory(ActionFactory, order=True))
