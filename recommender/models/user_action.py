# pylint: disable=missing-module-docstring, missing-class-docstring

from mongoengine import (
    Document,
    BooleanField,
    ReferenceField,
    DateTimeField,
    EmbeddedDocumentField,
    IntField,
)

from .action import Action
from .source import Source
from .target import Target
from .user import User


class UserAction(Document):
    logged_user = BooleanField()
    user = ReferenceField(User, blank=True)
    unique_id = IntField(blank=True)
    timestamp = DateTimeField()
    source = EmbeddedDocumentField(Source)
    target = EmbeddedDocumentField(Target)
    action = EmbeddedDocumentField(Action)
