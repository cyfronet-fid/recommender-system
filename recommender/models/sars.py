# pylint: disable=missing-module-docstring, missing-class-docstring

from mongoengine import (
    ListField,
    ReferenceField,
    Document,
    StringField,
    BooleanField,
)

from .recommendation import Recommendation
from .service import Service
from .state import State


class Sars(Document):
    state = ReferenceField(State)
    action = ListField(ReferenceField(Service))
    reward = ListField(ListField(StringField()))
    next_state = ReferenceField(State)
    synthetic = BooleanField(default=False)
    source_recommendation = ReferenceField(Recommendation, blank=True)
