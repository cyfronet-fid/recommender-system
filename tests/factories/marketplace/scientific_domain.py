# pylint: disable-all

from factory import LazyFunction
from faker import Factory as FakerFactory

from recommender.models import ScientificDomain
from .marketplace_document import MarketplaceDocument

from faker.providers import BaseProvider
from factory.random import reseed_random, random
from tests.factories.marketplace.faker_seeds.utils.loaders import load_names

reseed_random("test-seed")
fake = FakerFactory.create()


class ScientificDomainProvider(BaseProvider):
    SCIENTIFIC_DOMAIN_NAMES = load_names(ScientificDomain)

    def scientific_domain_name(self):
        return random.choice(list(self.SCIENTIFIC_DOMAIN_NAMES))


fake.add_provider(ScientificDomainProvider)


class ScientificDomainFactory(MarketplaceDocument):
    class Meta:
        model = ScientificDomain

    name = LazyFunction(lambda: fake.scientific_domain_name())
