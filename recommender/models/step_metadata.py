# pylint: disable=missing-module-docstring, missing-class-docstring
from enum import Enum, auto

from mongoengine import (
    StringField,
    DictField,
    EmbeddedDocument,
    EnumField,
    DateTimeField,
)


class Status(Enum):
    COMPLETED = auto()
    NOT_COMPLETED = auto()


class StepMetadata(EmbeddedDocument):
    type = StringField()
    start_time = DateTimeField()
    end_time = DateTimeField()
    details = DictField()
    status = EnumField(enum=Status)
