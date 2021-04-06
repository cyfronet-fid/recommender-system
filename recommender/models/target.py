# pylint: disable=missing-module-docstring, missing-class-docstring

from mongoengine import StringField, EmbeddedDocument, UUIDField


class Target(EmbeddedDocument):
    visit_id = UUIDField()
    page_id = StringField()
    meta = {"allow_inheritance": True}
