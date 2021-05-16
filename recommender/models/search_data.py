# pylint: disable=missing-module-docstring, missing-class-docstring
from mongoengine import (
    StringField,
    ReferenceField,
    ListField,
    Document,
)

from .category import Category
from .provider import Provider
from .platform import Platform
from .scientific_domain import ScientificDomain
from .target_user import TargetUser


class SearchData(Document):
    q = StringField(blank=True)
    categories = ListField(ReferenceField(Category), blank=True)
    geographical_availabilities = ListField(StringField(), blank=True)
    order_type = StringField(blank=True)
    providers = ListField(ReferenceField(Provider), blank=True)
    related_platforms = ListField(ReferenceField(Platform), blank=True)
    scientific_domains = ListField(ReferenceField(ScientificDomain), blank=True)
    sort = StringField(blank=True)
    target_users = ListField(ReferenceField(TargetUser), blank=True)
