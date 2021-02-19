# pylint: disable-all

from factory import LazyFunction
from faker import Factory as FakerFactory

from recommender.models import Provider
from .marketplace_document import MarketplaceDocument

faker = FakerFactory.create()


class ProviderFactory(MarketplaceDocument):
    class Meta:
        model = Provider

    name = LazyFunction(lambda: " ".join(faker.words(nb=2)))
