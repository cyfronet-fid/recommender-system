# pylint: disable=missing-module-docstring, missing-class-docstring

from mongoengine import Document, StringField, BinaryField


class PytorchModule(Document):
    name = StringField()
    description = StringField()
    module_bytes = BinaryField()
