# pylint: disable=missing-module-docstring, missing-class-docstring

from mongoengine import (
    StringField,
    ReferenceField,
    ListField,
    BooleanField,
)

from .access_mode import AccessMode
from .access_type import AccessType
from .category import Category
from .catalogue import Catalogue
from .life_cycle_status import LifeCycleStatus
from .marketplace_document import MarketplaceDocument
from .platform import Platform
from .provider import Provider
from .scientific_domain import ScientificDomain
from .target_user import TargetUser
from .trl import Trl


class Datasource(MarketplaceDocument):
    name = StringField()
    pid = StringField()
    status = StringField()
    description = StringField()
    tagline = StringField()
    countries = ListField(StringField())
    order_type = StringField()
    categories = ListField(ReferenceField(Category))
    catalogues = ListField(ReferenceField(Catalogue))
    providers = ListField(ReferenceField(Provider))
    resource_organisation = ReferenceField(Provider)
    scientific_domains = ListField(ReferenceField(ScientificDomain))
    platforms = ListField(ReferenceField(Platform))
    target_users = ListField(ReferenceField(TargetUser))
    access_modes = ListField(ReferenceField(AccessMode))
    access_types = ListField(ReferenceField(AccessType))
    trls = ListField(ReferenceField(Trl))
    life_cycle_statuses = ListField(ReferenceField(LifeCycleStatus))
    related_services = ListField(ReferenceField("Service"))
    required_services = ListField(ReferenceField("Service"))
    horizontal = BooleanField()
    standards = ListField(StringField())
    open_source_technologies = ListField(StringField())

    meta = {
        "indexes": [
            {
                "fields": ["$name", "$description", "$tagline"],
                "default_language": "english",
                "weights": {"title": 10, "content": 5, "tagline": 2},
            }
        ]
    }
