# pylint: disable-all

from factory import LazyFunction
from faker import Factory as FakerFactory

from recommender.models import AccessType
from .marketplace_document import MarketplaceDocument

faker = FakerFactory.create()


class AccessTypeFactory(MarketplaceDocument):
    class Meta:
        model = AccessType

    name = LazyFunction(lambda: " ".join(faker.words(nb=2)))
    description = faker.sentence(nb_words=30)
