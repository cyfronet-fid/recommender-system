# pylint: disable=missing-module-docstring, missing-class-docstring

from mongoengine import EmbeddedDocument, StringField, BooleanField


class Action(EmbeddedDocument):
    type = StringField()
    text = StringField()
    order = BooleanField()
