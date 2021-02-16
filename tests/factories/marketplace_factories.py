# pylint: disable-all

import os
from faker import Factory as FakerFactory
import factory.random
from app.db.mongo_models import *

factory.random.reseed_random(os.environ.get("TEST_SEED"))

faker = FakerFactory.create()
random = factory.random.random


class OnlyNameDocumentFactory(factory.mongoengine.MongoEngineFactory):
    name = factory.LazyFunction(lambda: " ".join(faker.words(nb=2)))


class CategoryFactory(OnlyNameDocumentFactory):
    class Meta:
        model = Category

    id = factory.Sequence(lambda n: f"{n}")


class ProviderFactory(OnlyNameDocumentFactory):
    class Meta:
        model = Provider

    id = factory.Sequence(lambda n: f"{n}")


class ScientificDomainFactory(OnlyNameDocumentFactory):
    class Meta:
        model = ScientificDomain

    id = factory.Sequence(lambda n: f"{n}")


class PlatformFactory(OnlyNameDocumentFactory):
    class Meta:
        model = Platform

    id = factory.Sequence(lambda n: f"{n}")


class NameAndDescriptionDocumentFactory(OnlyNameDocumentFactory):
    description = faker.sentence(nb_words=30)

    id = factory.Sequence(lambda n: f"{n}")


class TargetUserFactory(NameAndDescriptionDocumentFactory):
    class Meta:
        model = TargetUser

    id = factory.Sequence(lambda n: f"{n}")


class AccessModeFactory(NameAndDescriptionDocumentFactory):
    class Meta:
        model = AccessMode

    id = factory.Sequence(lambda n: f"{n}")


class AccessTypeFactory(NameAndDescriptionDocumentFactory):
    class Meta:
        model = AccessType

    id = factory.Sequence(lambda n: f"{n}")


class TrlFactory(NameAndDescriptionDocumentFactory):
    class Meta:
        model = Trl

    id = factory.Sequence(lambda n: f"{n}")


class LifeCycleStatusFactory(NameAndDescriptionDocumentFactory):
    class Meta:
        model = LifeCycleStatus

    id = factory.Sequence(lambda n: f"{n}")


class CategoriesAndScientificDomainsDocumentFactory(
    factory.mongoengine.MongoEngineFactory
):
    categories = factory.LazyFunction(
        lambda: CategoryFactory.create_batch(random.randint(2, 5))
    )
    scientific_domains = factory.LazyFunction(
        lambda: ScientificDomainFactory.create_batch(random.randint(2, 5))
    )


class ServiceFactory(
    CategoriesAndScientificDomainsDocumentFactory, NameAndDescriptionDocumentFactory
):
    class Meta:
        model = Service

    id = factory.Sequence(lambda n: f"{n}")

    def _tagline():
        n = random.randint(1, 7)
        tags = [" ".join(faker.words(nb=random.randint(1, 10))) for _ in range(n)]
        tagline = ", ".join(tags)
        return tagline

    tagline = factory.LazyFunction(_tagline)
    countries = factory.LazyFunction(
        lambda: [faker.country_code() for _ in range(random.randint(2, 5))]
    )
    providers = factory.LazyFunction(
        lambda: ProviderFactory.create_batch(random.randint(2, 5))
    )
    resource_organisation = factory.SubFactory(ProviderFactory)
    platforms = factory.LazyFunction(
        lambda: PlatformFactory.create_batch(random.randint(2, 5))
    )
    target_users = factory.LazyFunction(
        lambda: TargetUserFactory.create_batch(random.randint(2, 5))
    )
    access_modes = factory.LazyFunction(
        lambda: AccessModeFactory.create_batch(random.randint(2, 5))
    )
    access_types = factory.LazyFunction(
        lambda: AccessTypeFactory.create_batch(random.randint(2, 5))
    )
    trls = factory.LazyFunction(lambda: TrlFactory.create_batch(random.randint(2, 5)))
    life_cycle_statuses = factory.LazyFunction(
        lambda: LifeCycleStatusFactory.create_batch(random.randint(2, 5))
    )
    related_services = []
    required_services = []


class UserFactory(CategoriesAndScientificDomainsDocumentFactory):
    class Meta:
        model = User

    id = factory.Sequence(lambda n: f"{n}")
    accessed_services = factory.LazyFunction(
        lambda: ServiceFactory.create_batch(random.randint(0, 10))
    )
