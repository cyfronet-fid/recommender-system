# pylint: disable=missing-module-docstring, missing-class-docstring

from mongoengine import ListField, ReferenceField

from .marketplace_document import MarketplaceDocument
from .service import Service
from .user import User


class Project(MarketplaceDocument):
    user_id = ReferenceField(User)
    services = ListField(ReferenceField(Service))
