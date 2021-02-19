# pylint: disable-all

from factory import LazyFunction
from faker import Factory as FakerFactory

from recommender.models import Platform
from .marketplace_document import MarketplaceDocument

faker = FakerFactory.create()


class PlatformFactory(MarketplaceDocument):
    class Meta:
        model = Platform

    name = LazyFunction(lambda: " ".join(faker.words(nb=2)))
