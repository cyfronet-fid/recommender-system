# pylint: disable=missing-module-docstring, missing-class-docstring

from mongoengine import (
    ListField,
    ReferenceField,
    Document,
)

from .search_data import SearchData
from .service import Service
from .user import User


class State(Document):
    user = ReferenceField(User, blank=True)
    services_history = ListField(ReferenceField(Service))
    search_data = ReferenceField(SearchData)
