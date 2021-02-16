# pylint: disable-all

import os
import factory
from faker import Factory as FakerFactory
import factory.random
from datetime import datetime
import time
from app.db.mongo_models import Sars
from .marketplace_factories import ServiceFactory
from .state_factory import StateFactory

factory.random.reseed_random(os.environ.get("TEST_SEED"))

faker = FakerFactory.create()
random = factory.random.random


class SarsFactory(factory.mongoengine.MongoEngineFactory):
    class Meta:
        model = Sars

    state = factory.SubFactory(StateFactory)
    action = factory.LazyFunction(
        lambda: ServiceFactory.create_batch(random.randint(2, 3))
    )

    @factory.lazy_attribute
    def reward(self):
        def _reward():
            return "_".join(faker.words(nb=random.randint(2, 6)))

        def _rewards_list():
            return [_reward() for _ in range(random.randint(3, 10))]

        n = len(self.action)
        return [_rewards_list() for _ in range(n)]

    @factory.lazy_attribute
    def next_state(self):
        return StateFactory(
            user=self.state.user,
            services_history=self.state.services_history + self.action,
            last_searchphrase=self.state.last_searchphrase,
            last_filters=self.state.last_filters,
        )
