# pylint: disable=missing-module-docstring, missing-class-docstring

from mongoengine import (
    ListField,
    ReferenceField,
    Document,
    BooleanField,
)

from .search_data import SearchData
from .service import Service
from .user import User


class State(Document):
    user = ReferenceField(User, blank=True)
    services_history = ListField(ReferenceField(Service))
    elastic_services = ListField(ReferenceField(Service))
    search_data = ReferenceField(SearchData)
    synthetic = BooleanField(default=False)
