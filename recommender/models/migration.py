# pylint: disable=missing-module-docstring, missing-class-docstring

from mongoengine import Document, BooleanField, StringField


class Migration(Document):
    name = StringField(required=True)
    applied = BooleanField(required=True)
