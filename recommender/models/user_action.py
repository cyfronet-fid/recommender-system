# pylint: disable=missing-module-docstring, missing-class-docstring

from mongoengine import (
    Document,
    ReferenceField,
    DateTimeField,
    EmbeddedDocumentField,
    UUIDField,
    BooleanField,
    StringField,
)

from .action import Action
from .source import Source
from .target import Target
from .user import User


class UserAction(Document):
    user = ReferenceField(User, blank=True)
    unique_id = UUIDField()
    timestamp = DateTimeField()
    client_id = StringField(blank=True)
    source = EmbeddedDocumentField(Source)
    target = EmbeddedDocumentField(Target)
    action = EmbeddedDocumentField(Action)
    processed = BooleanField(default=False)

    meta = {
        "indexes": [
            ("source.visit_id", "source.root.service"),
            "user",
            "unique_id",
            "timestamp",
            "source.root.type",
        ]
    }
