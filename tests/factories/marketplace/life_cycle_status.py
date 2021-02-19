# pylint: disable-all

from factory import LazyFunction, LazyAttribute
from faker import Factory as FakerFactory

from recommender.models import LifeCycleStatus
from .marketplace_document import MarketplaceDocument

from faker.providers import BaseProvider
from factory.random import reseed_random, random
from tests.factories.marketplace.faker_seeds.utils.loaders import load_names_descs

reseed_random("test-seed")
fake = FakerFactory.create()


class LifeCycleStatusProvider(BaseProvider):
    LIFE_CYCLE_NAMES_DESCS = load_names_descs(LifeCycleStatus)

    def life_cycle_status_name(self):
        return random.choice(list(self.LIFE_CYCLE_NAMES_DESCS.keys()))

    def life_cycle_status_description(self, name):
        return self.LIFE_CYCLE_NAMES_DESCS.get(name)


fake.add_provider(LifeCycleStatusProvider)


class LifeCycleStatusFactory(MarketplaceDocument):
    class Meta:
        model = LifeCycleStatus

    name = LazyFunction(lambda: fake.life_cycle_status_name())
    description = LazyAttribute(lambda o: fake.life_cycle_status_description(o.name))
