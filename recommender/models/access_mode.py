# pylint: disable=missing-module-docstring, missing-class-docstring

from mongoengine import StringField

from .marketplace_document import MarketplaceDocument


class AccessMode(MarketplaceDocument):
    name = StringField()
    description = StringField()
