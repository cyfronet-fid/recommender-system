# pylint: disable-all

import os
from faker import Factory as FakerFactory
import factory.random
from app.db.mongo_models import State
from .marketplace_factories import UserFactory, ServiceFactory

faker = FakerFactory.create()
random = factory.random.random

factory.random.reseed_random(os.environ.get("TEST_SEED"))


class StateFactory(factory.mongoengine.MongoEngineFactory):
    class Meta:
        model = State

    user = factory.SubFactory(UserFactory)
    services_history = factory.LazyFunction(
        lambda: ServiceFactory.create_batch(random.randint(0, 10))
    )
    last_searchphrase = factory.LazyFunction(
        lambda: " ".join(faker.words(nb=random.randint(2, 10)))
    )
    last_filters = factory.LazyFunction(lambda: faker.words(nb=random.randint(2, 10)))
