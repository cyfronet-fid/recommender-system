import os

from pymodm import connect
from inflection import underscore, pluralize

from app.db.mongo_models import *


class DbHandler:
    def __init__(self, db_name=os.environ["MONGO_DB_NAME"]):
        connect(os.environ["MONGO_DB_ENDPOINT"] + db_name)

        self.mongo_model_classes = [
            Category,
            Provider,
            ScientificDomain,
            Platform,
            TargetUser,
            AccessMode,
            AccessType,
            Trl,
            LifeCycleStatus,
            User,
            Service,
        ]

    def load(self, data):
        for model in self.mongo_model_classes:
            json_key = pluralize(underscore(model.__name__))
            model.objects.bulk_create([model(**x) for x in data[json_key]])

    def drop(self):
        for model in self.mongo_model_classes:
            model.objects.all().delete()
