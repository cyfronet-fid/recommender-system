# pylint: disable=missing-module-docstring, missing-class-docstring

from mongoengine import (
    Document,
    ReferenceField,
    DateTimeField,
    IntField,
    StringField,
    ListField,
    EmbeddedDocumentField,
)

from .user import User
from .service import Service
from .search_data import SearchData


class Recommendation(Document):
    user = ReferenceField(User, blank=True)
    unique_id = IntField()
    timestamp = DateTimeField()
    visit_id = IntField()
    page_id = StringField()
    panel_id = StringField()
    services = ListField(ReferenceField(Service))
    search_data = EmbeddedDocumentField(SearchData)
