# pylint: disable-all

import time
from factory import SubFactory, LazyFunction, Trait
from factory.mongoengine import MongoEngineFactory
from factory.random import reseed_random, random
from faker import Factory as FakerFactory
from datetime import datetime

from recommender.models import Recommendation
from .marketplace import UserFactory, ServiceFactory

reseed_random("test-seed")
faker = FakerFactory.create()


class RecommendationFactory(MongoEngineFactory):
    class Meta:
        model = Recommendation

    logged_user = True
    user = SubFactory(UserFactory)
    unique_id = None
    timestamp = LazyFunction(lambda: datetime.fromtimestamp(time.time()))
    visit_id = LazyFunction(lambda: faker.uuid4(cast_to=None).int >> 65)
    page_id = LazyFunction(lambda: "_".join(faker.words(nb=random.randint(2, 6))))
    panel_id = LazyFunction(lambda: random.choice(["version_A", "version_B"]))

    services = LazyFunction(lambda: ServiceFactory.create_batch(random.randint(2, 10)))
    search_phrase = LazyFunction(
        lambda: " ".join(faker.words(nb=random.randint(2, 10)))
    )
    filters = LazyFunction(lambda: faker.words(nb=random.randint(2, 10)))

    class Params:
        version_A = Trait(
            services=LazyFunction(lambda: ServiceFactory.create_batch(3)),
            panel_id="version_A",
        )
        version_1 = version_A

        version_B = Trait(
            services=LazyFunction(lambda: ServiceFactory.create_batch(2)),
            panel_id="version_B",
        )
        version_2 = version_B

        not_logged = Trait(
            logged_user=False,
            user=None,
            unique_id=LazyFunction(lambda: faker.uuid4(cast_to=None).int >> 65),
        )
