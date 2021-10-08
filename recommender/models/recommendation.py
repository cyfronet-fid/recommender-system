# pylint: disable=missing-module-docstring, missing-class-docstring

from mongoengine import (
    Document,
    ReferenceField,
    DateTimeField,
    StringField,
    ListField,
    UUIDField,
)

from .user import User
from .service import Service
from .search_data import SearchData


class Recommendation(Document):
    user = ReferenceField(User, blank=True)
    unique_id = UUIDField()
    timestamp = DateTimeField()
    visit_id = UUIDField()
    page_id = StringField()
    panel_id = StringField()
    engine_version = StringField()
    services = ListField(ReferenceField(Service))
    search_data = ReferenceField(SearchData)
