# pylint: disable=missing-module-docstring, missing-class-docstring

from mongoengine import Document, BinaryField, StringField


class MLComponent(Document):
    type = StringField()
    version = StringField()
    binary_object = BinaryField()
