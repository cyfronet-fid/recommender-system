# pylint: disable-all

from factory import LazyFunction
from factory.mongoengine import MongoEngineFactory
from factory.random import reseed_random, random
from faker import Factory as FakerFactory

from recommender.models import Target

faker = FakerFactory.create()
reseed_random("test-seed")


class TargetFactory(MongoEngineFactory):
    class Meta:
        model = Target

    visit_id = LazyFunction(lambda: faker.uuid4(cast_to=None))
    page_id = LazyFunction(lambda: "_".join(faker.words(nb=random.randint(2, 6))))
