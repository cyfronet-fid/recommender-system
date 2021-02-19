# pylint: disable-all

from factory import LazyFunction, LazyAttribute
from faker import Factory as FakerFactory

from recommender.models import AccessMode
from .marketplace_document import MarketplaceDocument

from faker.providers import BaseProvider
from factory.random import reseed_random, random
from tests.factories.marketplace.faker_seeds.utils.loaders import load_names_descs

reseed_random("test-seed")
fake = FakerFactory.create()


class AccessModeProvider(BaseProvider):
    ACCESS_MODE_NAMES_DESCS = load_names_descs(AccessMode)

    def access_mode_name(self):
        return random.choice(list(self.ACCESS_MODE_NAMES_DESCS.keys()))

    def access_mode_description(self, name):
        return self.ACCESS_MODE_NAMES_DESCS.get(name)


fake.add_provider(AccessModeProvider)


class AccessModeFactory(MarketplaceDocument):
    class Meta:
        model = AccessMode

    name = LazyFunction(lambda: fake.access_mode_name())
    description = LazyAttribute(lambda o: fake.access_mode_description(o.name))
