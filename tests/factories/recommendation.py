# pylint: disable-all

import time
from factory import SubFactory, LazyFunction, Trait
from factory.mongoengine import MongoEngineFactory
from factory.random import reseed_random, random
from faker import Factory as FakerFactory
from datetime import datetime

from recommender.models import Recommendation
from .marketplace import UserFactory, ServiceFactory
from .search_data import SearchDataFactory

reseed_random("test-seed")
faker = FakerFactory.create()


class RecommendationFactory(MongoEngineFactory):
    class Meta:
        model = Recommendation

    user = SubFactory(UserFactory)
    unique_id = LazyFunction(lambda: faker.uuid4())
    timestamp = LazyFunction(lambda: datetime.fromtimestamp(time.time()))
    visit_id = LazyFunction(lambda: faker.uuid4(cast_to=None))
    page_id = LazyFunction(lambda: "_".join(faker.words(nb=random.randint(2, 6))))
    panel_id = LazyFunction(lambda: random.choice(["v1", "v2"]))
    engine_version = LazyFunction(lambda: random.choice(["pre_agent", "rl_agent"]))
    services = LazyFunction(lambda: ServiceFactory.create_batch(random.randint(2, 10)))
    search_data = SubFactory(SearchDataFactory)

    class Params:
        v1 = Trait(
            services=LazyFunction(lambda: ServiceFactory.create_batch(3)),
            panel_id="v1",
        )

        v2 = Trait(
            services=LazyFunction(lambda: ServiceFactory.create_batch(2)),
            panel_id="v2",
        )

        not_logged = Trait(user=None)
