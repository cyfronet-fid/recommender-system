# pylint: disable=missing-module-docstring, missing-class-docstring

from mongoengine import (
    ListField,
    ReferenceField,
    BinaryField,
    FloatField,
    BooleanField,
    IntField,
)

from .category import Category
from .marketplace_document import MarketplaceDocument
from .scientific_domain import ScientificDomain
from .service import Service


class User(MarketplaceDocument):
    scientific_domains = ListField(ReferenceField(ScientificDomain))
    categories = ListField(ReferenceField(Category))
    accessed_services = ListField(ReferenceField(Service))
    synthetic = BooleanField(default=False)
    dataframe = BinaryField(blank=True)
    one_hot_tensor = ListField(IntField(), blank=True)
    dense_tensor = ListField(FloatField(), blank=True)

    meta = {"indexes": ["id"]}
