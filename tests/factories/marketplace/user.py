# pylint: disable-all

from factory import LazyFunction
from factory.random import reseed_random, random
from faker import Factory as FakerFactory

from recommender.models import User
from .service import ServiceFactory
from .marketplace_document import MarketplaceDocument
from .scientific_domain import ScientificDomainFactory
from .category import CategoryFactory

faker = FakerFactory.create()
reseed_random("test-seed")


class UserFactory(MarketplaceDocument):
    class Meta:
        model = User

    categories = LazyFunction(
        lambda: CategoryFactory.create_batch(random.randint(2, 5))
    )
    scientific_domains = LazyFunction(
        lambda: ScientificDomainFactory.create_batch(random.randint(2, 5))
    )
    accessed_services = LazyFunction(
        lambda: ServiceFactory.create_batch(random.randint(0, 10))
    )
