# pylint: disable=missing-module-docstring, missing-class-docstring

from mongoengine import EmbeddedDocument, StringField, ReferenceField

from .service import Service


class Root(EmbeddedDocument):
    type = StringField()
    panel_id = StringField(blank=True)
    service = ReferenceField(Service)
