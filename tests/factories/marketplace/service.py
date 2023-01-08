# pylint: disable-all

from factory import LazyFunction, LazyAttribute, SubFactory
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

from faker.providers import BaseProvider
from factory.random import reseed_random, random
from tests.factories.marketplace.faker_seeds.utils.loaders import (
    load_names_descs,
    load_taglines,
)

reseed_random("test-seed")
fake = FakerFactory.create()


class ServiceProvider(BaseProvider):
    SERVICE_NAMES_DESCS = load_names_descs(Service)
    TAGLINES = load_taglines()

    def service_name(self):
        return random.choice(list(self.SERVICE_NAMES_DESCS.keys()))

    def service_description(self, name):
        return self.SERVICE_NAMES_DESCS.get(name)

    def service_tagline(self):
        return random.choice(self.TAGLINES)


fake.add_provider(ServiceProvider)


class ServiceFactory(MarketplaceDocument):
    class Meta:
        model = Service

    name = LazyFunction(lambda: fake.service_name())
    description = LazyAttribute(lambda o: fake.service_description(o.name))
    tagline = LazyFunction(lambda: fake.service_tagline())
    countries = LazyFunction(
        lambda: [fake.country_code() for _ in range(random.randint(2, 5))]
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
    status = "published"
    pid = "pid"
