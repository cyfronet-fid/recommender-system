# pylint: disable-all

from factory import LazyFunction
from faker import Factory as FakerFactory

from recommender.models import Provider
from .marketplace_document import MarketplaceDocument

from faker.providers import BaseProvider
from factory.random import reseed_random, random
from tests.factories.marketplace.faker_seeds.utils.loaders import load_names

reseed_random("test-seed")
fake = FakerFactory.create()


class ProviderProvider(BaseProvider):
    PROVIDER_NAMES = load_names(Provider)

    def provider_name(self):
        return random.choice(list(self.PROVIDER_NAMES))


fake.add_provider(ProviderProvider)


class ProviderFactory(MarketplaceDocument):
    class Meta:
        model = Provider

    name = LazyFunction(lambda: fake.provider_name())
