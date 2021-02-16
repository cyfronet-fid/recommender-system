# pylint: disable-all

import os
from faker import Factory as FakerFactory
import factory.random
from app.db.mongo_models import Target

faker = FakerFactory.create()
random = factory.random.random

factory.random.reseed_random(os.environ.get("TEST_SEED"))


class TargetFactory(factory.mongoengine.MongoEngineFactory):
    class Meta:
        model = Target

    visit_id = factory.LazyFunction(lambda: faker.uuid4(cast_to=None).int >> 65)
    page_id = factory.LazyFunction(
        lambda: "_".join(faker.words(nb=random.randint(2, 6)))
    )
