# pylint: disable=missing-module-docstring, missing-class-docstring

from mongoengine import IntField, StringField, EmbeddedDocument


class Target(EmbeddedDocument):
    visit_id = IntField()
    page_id = StringField()
    meta = {"allow_inheritance": True}
