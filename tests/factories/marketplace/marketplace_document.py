# pylint: disable-all

from factory import Sequence
from factory.mongoengine import MongoEngineFactory
from faker import Factory as FakerFactory

faker = FakerFactory.create()


class MarketplaceDocument(MongoEngineFactory):
    id = Sequence(lambda n: f"{n}")
