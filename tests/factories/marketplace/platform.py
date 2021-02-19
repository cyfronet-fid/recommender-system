# pylint: disable-all

from factory import LazyFunction
from faker import Factory as FakerFactory

from recommender.models import Platform
from .marketplace_document import MarketplaceDocument

from faker.providers import BaseProvider
from factory.random import reseed_random, random
from tests.factories.marketplace.faker_seeds.utils.loaders import load_names

reseed_random("test-seed")
fake = FakerFactory.create()


class PlatformProvider(BaseProvider):
    PLATFORM_NAMES = load_names(Platform)

    def platform_name(self):
        return random.choice(list(self.PLATFORM_NAMES))


fake.add_provider(PlatformProvider)


class PlatformFactory(MarketplaceDocument):
    class Meta:
        model = Platform

    name = LazyFunction(lambda: fake.platform_name())
