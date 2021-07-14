# pylint: disable-all

from factory import SubFactory, LazyFunction, Trait
from factory.mongoengine import MongoEngineFactory
from factory.random import reseed_random, random
from faker import Factory as FakerFactory

from recommender.models import State
from .marketplace import UserFactory, ServiceFactory
from .search_data import SearchDataFactory

faker = FakerFactory.create()
reseed_random("test-seed")


class StateFactory(MongoEngineFactory):
    class Meta:
        model = State

    user = SubFactory(UserFactory)
    services_history = LazyFunction(
        lambda: ServiceFactory.create_batch(random.randint(0, 10))
    )
    search_data = SubFactory(SearchDataFactory)

    class Params:
        non_empty_history = Trait(
            services_history=LazyFunction(
                lambda: ServiceFactory.create_batch(random.randint(3, 10))
            )
        )
