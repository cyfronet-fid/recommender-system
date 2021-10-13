# pylint: disable=missing-module-docstring, missing-class-docstring
from enum import Enum, auto

from mongoengine import StringField, DateField, DictField, EmbeddedDocument, EnumField


class Status(Enum):
    COMPLETED = auto()
    NOT_COMPLETED = auto()


class StepMetadata(EmbeddedDocument):
    type = StringField()
    start_time = DateField()
    end_time = DateField()
    details = DictField()
    status = EnumField(enum=Status)
