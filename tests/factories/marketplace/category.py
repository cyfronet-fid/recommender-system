# pylint: disable-all

from factory import LazyFunction
from faker import Factory as FakerFactory

from recommender.models import Category
from .marketplace_document import MarketplaceDocument

from faker.providers import BaseProvider
from factory.random import reseed_random, random
from tests.factories.marketplace.faker_seeds.utils.loaders import load_names

reseed_random("test-seed")
fake = FakerFactory.create()


class CategoryProvider(BaseProvider):
    CATEGORY_NAMES = load_names(Category)

    def category_name(self):
        return random.choice(list(self.CATEGORY_NAMES))


fake.add_provider(CategoryProvider)


class CategoryFactory(MarketplaceDocument):
    class Meta:
        model = Category

    name = LazyFunction(lambda: fake.category_name())
