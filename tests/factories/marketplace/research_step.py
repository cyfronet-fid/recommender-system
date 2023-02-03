# pylint: disable-all

from factory import LazyFunction, LazyAttribute
from faker import Factory as FakerFactory

from recommender.models import ResearchStep
from .marketplace_document import MarketplaceDocument

from faker.providers import BaseProvider
from factory.random import reseed_random, random
from tests.factories.marketplace.faker_seeds.utils.loaders import load_names_descs

reseed_random("test-seed")
fake = FakerFactory.create()


class ResearchStepProvider(BaseProvider):
    RESEARCH_STEP_NAMES_DESCS = load_names_descs(ResearchStep)

    def research_step_name(self):
        return random.choice(list(self.RESEARCH_STEP_NAMES_DESCS.keys()))

    def research_step_description(self, name):
        return self.RESEARCH_STEP_NAMES_DESCS.get(name)


fake.add_provider(ResearchStepProvider)


class ResearchStepFactory(MarketplaceDocument):
    class Meta:
        model = ResearchStep

    name = LazyFunction(lambda: fake.research_step_name())
    description = LazyAttribute(lambda o: fake.research_step_description(o.name))
