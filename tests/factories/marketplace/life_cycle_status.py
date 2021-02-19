# pylint: disable-all

from factory import LazyFunction
from faker import Factory as FakerFactory

from recommender.models import LifeCycleStatus
from .marketplace_document import MarketplaceDocument

faker = FakerFactory.create()


class LifeCycleStatusFactory(MarketplaceDocument):
    class Meta:
        model = LifeCycleStatus

    name = LazyFunction(lambda: " ".join(faker.words(nb=2)))
    description = faker.sentence(nb_words=30)
