# pylint: disable-all

from factory import LazyFunction, LazyAttribute
from faker import Factory as FakerFactory

from recommender.models import Trl
from .marketplace_document import MarketplaceDocument

from faker.providers import BaseProvider
from factory.random import reseed_random, random
from tests.factories.marketplace.faker_seeds.utils.loaders import load_names_descs

reseed_random("test-seed")
fake = FakerFactory.create()


class TrlProvider(BaseProvider):
    TRL_NAMES_DESCS = load_names_descs(Trl)

    def trl_name(self):
        return random.choice(list(self.TRL_NAMES_DESCS.keys()))

    def trl_description(self, name):
        return self.TRL_NAMES_DESCS.get(name)


fake.add_provider(TrlProvider)


class TrlFactory(MarketplaceDocument):
    class Meta:
        model = Trl

    name = LazyFunction(lambda: fake.trl_name())
    description = LazyAttribute(lambda o: fake.trl_description(o.name))
