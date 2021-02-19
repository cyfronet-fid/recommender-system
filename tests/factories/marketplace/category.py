# pylint: disable-all

from factory import LazyFunction
from faker import Factory as FakerFactory

from recommender.models import Category
from .marketplace_document import MarketplaceDocument

faker = FakerFactory.create()


class CategoryFactory(MarketplaceDocument):
    class Meta:
        model = Category

    name = LazyFunction(lambda: " ".join(faker.words(nb=2)))
