# pylint: disable=missing-module-docstring, missing-class-docstring

from mongoengine import (
    ListField,
    ReferenceField,
    Document,
    EmbeddedDocumentField,
    StringField,
)

from .service import Service
from .state import State


class Sars(Document):
    state = EmbeddedDocumentField(State)
    action = ListField(ReferenceField(Service))
    reward = ListField(ListField(StringField()))
    next_state = EmbeddedDocumentField(State)
