# pylint: disable=missing-module-docstring, missing-class-docstring

from mongoengine import (
    Document,
    ReferenceField,
    DateTimeField,
    StringField,
    ListField,
    EmbeddedDocumentField,
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
    services = ListField(ReferenceField(Service))
    search_data = EmbeddedDocumentField(SearchData)
