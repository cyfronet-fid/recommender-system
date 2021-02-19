# pylint: disable=missing-module-docstring, missing-class-docstring

from mongoengine import EmbeddedDocument, ListField, ReferenceField, StringField

from .service import Service
from .user import User


class State(EmbeddedDocument):
    user = ReferenceField(User, blank=True)
    services_history = ListField(ReferenceField(Service))
    last_searchphrase = StringField()
    last_filters = ListField(StringField())
