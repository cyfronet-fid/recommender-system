# pylint: disable-all

from factory import LazyFunction
from faker import Factory as FakerFactory

from recommender.models import ScientificDomain
from .marketplace_document import MarketplaceDocument

faker = FakerFactory.create()


class ScientificDomainFactory(MarketplaceDocument):
    class Meta:
        model = ScientificDomain

    name = LazyFunction(lambda: " ".join(faker.words(nb=2)))
