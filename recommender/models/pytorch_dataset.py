# pylint: disable=missing-module-docstring, missing-class-docstring

from mongoengine import Document, StringField, BinaryField


class PytorchDataset(Document):
    name = StringField()
    description = StringField()
    dataset_bytes = BinaryField()
