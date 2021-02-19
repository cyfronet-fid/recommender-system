# pylint: disable-all

from factory import SubFactory, LazyFunction
from factory.mongoengine import MongoEngineFactory
from factory.random import reseed_random, random
from faker import Factory as FakerFactory

from recommender.models import State
from .marketplace import UserFactory, ServiceFactory

faker = FakerFactory.create()
reseed_random("test-seed")


class StateFactory(MongoEngineFactory):
    class Meta:
        model = State

    user = SubFactory(UserFactory)
    services_history = LazyFunction(
        lambda: ServiceFactory.create_batch(random.randint(0, 10))
    )
    last_searchphrase = LazyFunction(
        lambda: " ".join(faker.words(nb=random.randint(2, 10)))
    )
    last_filters = LazyFunction(lambda: faker.words(nb=random.randint(2, 10)))
