# pylint: disable-all

from factory import SubFactory, LazyFunction, lazy_attribute, Trait
from factory.mongoengine import MongoEngineFactory
from factory.random import reseed_random, random
from faker import Factory as FakerFactory

from recommender.models import Sars
from .marketplace import ServiceFactory
from .state import StateFactory

faker = FakerFactory.create()
reseed_random("test-seed")


class SarsFactory(MongoEngineFactory):
    class Meta:
        model = Sars

    state = SubFactory(StateFactory)
    action = LazyFunction(lambda: ServiceFactory.create_batch(random.randint(2, 3)))

    @lazy_attribute
    def reward(self):
        def _reward():
            return "_".join(faker.words(nb=random.randint(2, 6)))

        def _rewards_list():
            return [_reward() for _ in range(random.randint(3, 10))]

        n = len(self.action)
        return [_rewards_list() for _ in range(n)]

    @lazy_attribute
    def next_state(self):
        return StateFactory(
            user=self.state.user,
            services_history=self.state.services_history + self.action,
            last_search_data=self.state.last_search_data,
        )

    class Params:
        K_2 = Trait(action=LazyFunction(lambda: ServiceFactory.create_batch(2)))
        K_3 = Trait(action=LazyFunction(lambda: ServiceFactory.create_batch(3)))
