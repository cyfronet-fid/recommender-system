# pylint: disable-all

from factory import LazyFunction
from faker import Factory as FakerFactory

from recommender.models import TargetUser
from .marketplace_document import MarketplaceDocument

faker = FakerFactory.create()


class TargetUserFactory(MarketplaceDocument):
    class Meta:
        model = TargetUser

    name = LazyFunction(lambda: " ".join(faker.words(nb=2)))
    description = faker.sentence(nb_words=30)
