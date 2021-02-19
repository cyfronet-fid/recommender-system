# pylint: disable-all

from factory import LazyFunction, SubFactory
from factory.random import reseed_random, random
from faker import Factory as FakerFactory

from recommender.models import Service
from .access_mode import AccessModeFactory
from .access_type import AccessTypeFactory
from .category import CategoryFactory
from .life_cycle_status import LifeCycleStatusFactory
from .marketplace_document import MarketplaceDocument
from .platform import PlatformFactory
from .provider import ProviderFactory
from .scientific_domain import ScientificDomainFactory
from .target_user import TargetUserFactory
from .trl import TrlFactory

faker = FakerFactory.create()
reseed_random("test-seed")


def _tagline():
    n = random.randint(1, 7)
    tags = [" ".join(faker.words(nb=random.randint(1, 10))) for _ in range(n)]
    tagline = ", ".join(tags)
    return tagline


class ServiceFactory(MarketplaceDocument):
    class Meta:
        model = Service

    name = LazyFunction(lambda: " ".join(faker.words(nb=2)))
    description = faker.sentence(nb_words=30)
    tagline = LazyFunction(_tagline)
    countries = LazyFunction(
        lambda: [faker.country_code() for _ in range(random.randint(2, 5))]
    )
    providers = LazyFunction(lambda: ProviderFactory.create_batch(random.randint(2, 5)))
    resource_organisation = SubFactory(ProviderFactory)
    platforms = LazyFunction(lambda: PlatformFactory.create_batch(random.randint(2, 5)))
    target_users = LazyFunction(
        lambda: TargetUserFactory.create_batch(random.randint(2, 5))
    )
    access_modes = LazyFunction(
        lambda: AccessModeFactory.create_batch(random.randint(2, 5))
    )
    access_types = LazyFunction(
        lambda: AccessTypeFactory.create_batch(random.randint(2, 5))
    )
    trls = LazyFunction(lambda: TrlFactory.create_batch(random.randint(2, 5)))
    life_cycle_statuses = LazyFunction(
        lambda: LifeCycleStatusFactory.create_batch(random.randint(2, 5))
    )
    categories = LazyFunction(
        lambda: CategoryFactory.create_batch(random.randint(2, 5))
    )
    scientific_domains = LazyFunction(
        lambda: ScientificDomainFactory.create_batch(random.randint(2, 5))
    )
    related_services = []
    required_services = []
