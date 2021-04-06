# pylint: disable=missing-module-docstring, missing-class-docstring

from mongoengine import (
    Document,
    ReferenceField,
    DateTimeField,
    EmbeddedDocumentField,
    UUIDField,
)

from .action import Action
from .source import Source
from .target import Target
from .user import User


class UserAction(Document):
    user = ReferenceField(User, blank=True)
    unique_id = UUIDField()
    timestamp = DateTimeField()
    source = EmbeddedDocumentField(Source)
    target = EmbeddedDocumentField(Target)
    action = EmbeddedDocumentField(Action)
