# pylint: disable=missing-module-docstring, missing-class-docstring

from mongoengine import Document, IntField


class MarketplaceDocument(Document):
    id = IntField(primary_key=True)
    meta = {
        "allow_inheritance": True,
        "abstract": True,
    }
