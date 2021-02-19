# pylint: disable-all

from factory import LazyFunction, LazyAttribute
from faker import Factory as FakerFactory

from recommender.models import TargetUser
from .marketplace_document import MarketplaceDocument

from faker.providers import BaseProvider
from factory.random import reseed_random, random
from tests.factories.marketplace.faker_seeds.utils.loaders import load_names_descs

reseed_random("test-seed")
fake = FakerFactory.create()


class TargetUserProvider(BaseProvider):
    TARGET_USER_NAMES_DESCS = load_names_descs(TargetUser)

    def target_user_name(self):
        return random.choice(list(self.TARGET_USER_NAMES_DESCS.keys()))

    def target_user_description(self, name):
        return self.TARGET_USER_NAMES_DESCS.get(name)


fake.add_provider(TargetUserProvider)


class TargetUserFactory(MarketplaceDocument):
    class Meta:
        model = TargetUser

    name = LazyFunction(lambda: fake.target_user_name())
    description = LazyAttribute(lambda o: fake.target_user_description(o.name))
