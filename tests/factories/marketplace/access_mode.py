# pylint: disable-all

from factory import LazyFunction
from faker import Factory as FakerFactory

from recommender.models import AccessMode
from .marketplace_document import MarketplaceDocument

faker = FakerFactory.create()


class AccessModeFactory(MarketplaceDocument):
    class Meta:
        model = AccessMode

    name = LazyFunction(lambda: " ".join(faker.words(nb=2)))
    description = faker.sentence(nb_words=30)
