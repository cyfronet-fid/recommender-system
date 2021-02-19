# pylint: disable-all

from factory.mongoengine import MongoEngineFactory

from recommender.models import Action


class ActionFactory(MongoEngineFactory):
    class Meta:
        model = Action

    type = "button"
    text = "Details"
    order = False
