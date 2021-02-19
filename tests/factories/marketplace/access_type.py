# pylint: disable-all

from factory import LazyFunction, LazyAttribute
from faker import Factory as FakerFactory

from recommender.models import AccessType
from .marketplace_document import MarketplaceDocument

from faker.providers import BaseProvider
from factory.random import reseed_random, random
from tests.factories.marketplace.faker_seeds.utils.loaders import load_names_descs

reseed_random("test-seed")
fake = FakerFactory.create()


class AccessTypeProvider(BaseProvider):
    ACCESS_TYPE_NAMES_DESCS = load_names_descs(AccessType)

    def access_type_name(self):
        return random.choice(list(self.ACCESS_TYPE_NAMES_DESCS.keys()))

    def access_type_description(self, name):
        return self.ACCESS_TYPE_NAMES_DESCS.get(name)


fake.add_provider(AccessTypeProvider)


class AccessTypeFactory(MarketplaceDocument):
    class Meta:
        model = AccessType

    name = LazyFunction(lambda: fake.access_type_name())
    description = LazyAttribute(lambda o: fake.access_type_description(o.name))
