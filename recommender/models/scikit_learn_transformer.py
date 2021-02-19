# pylint: disable=missing-module-docstring, missing-class-docstring

from mongoengine import Document, StringField, BinaryField


class ScikitLearnTransformer(Document):
    name = StringField()
    description = StringField()
    binary_transformer = BinaryField()
