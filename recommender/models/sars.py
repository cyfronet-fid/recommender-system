# pylint: disable=missing-module-docstring, missing-class-docstring

from mongoengine import (
    ListField,
    ReferenceField,
    Document,
    StringField,
)

from .service import Service
from .state import State


class Sars(Document):
    state = ReferenceField(State)
    action = ListField(ReferenceField(Service))
    reward = ListField(ListField(StringField()))
    next_state = ReferenceField(State)
