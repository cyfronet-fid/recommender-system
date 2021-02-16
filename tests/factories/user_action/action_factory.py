# pylint: disable-all

import factory
from app.db.mongo_models import Action


class ActionFactory(factory.mongoengine.MongoEngineFactory):
    class Meta:
        model = Action

    type = "button"
    text = "Details"
    order = False
