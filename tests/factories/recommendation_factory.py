# pylint: disable-all

import os
from faker import Factory as FakerFactory
import factory.random
from datetime import datetime
import time
from app.db.mongo_models import Recommendation
from .marketplace_factories import UserFactory, ServiceFactory

factory.random.reseed_random(os.environ.get("TEST_SEED"))

faker = FakerFactory.create()
random = factory.random.random


class RecommendationFactory(factory.mongoengine.MongoEngineFactory):
    class Meta:
        model = Recommendation

    logged_user = True
    user = factory.SubFactory(UserFactory)
    unique_id = None
    timestamp = factory.LazyFunction(lambda: datetime.fromtimestamp(time.time()))
    visit_id = factory.LazyFunction(lambda: faker.uuid4(cast_to=None).int >> 65)
    page_id = factory.LazyFunction(
        lambda: "_".join(faker.words(nb=random.randint(2, 6)))
    )
    panel_id = factory.LazyFunction(lambda: random.choice(["version_A", "version_B"]))

    services = factory.LazyFunction(
        lambda: ServiceFactory.create_batch(random.randint(2, 10))
    )
    search_phrase = factory.LazyFunction(
        lambda: " ".join(faker.words(nb=random.randint(2, 10)))
    )
    filters = factory.LazyFunction(lambda: faker.words(nb=random.randint(2, 10)))

    class Params:
        version_A = factory.Trait(
            services=factory.LazyFunction(lambda: ServiceFactory.create_batch(3)),
            panel_id="version_A",
        )
        version_1 = version_A

        version_B = factory.Trait(
            services=factory.LazyFunction(lambda: ServiceFactory.create_batch(2)),
            panel_id="version_B",
        )
        version_2 = version_B

        not_logged = factory.Trait(
            logged_user=False,
            user=None,
            unique_id=factory.LazyFunction(lambda: faker.uuid4(cast_to=None).int >> 65),
        )
