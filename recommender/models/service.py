# pylint: disable=missing-module-docstring, missing-class-docstring

from mongoengine import (
    StringField,
    ReferenceField,
    ListField,
    BinaryField,
    FloatField,
    IntField,
    BooleanField,
)

from .access_mode import AccessMode
from .access_type import AccessType
from .category import Category
from .life_cycle_status import LifeCycleStatus
from .marketplace_document import MarketplaceDocument
from .platform import Platform
from .provider import Provider
from .scientific_domain import ScientificDomain
from .target_user import TargetUser
from .trl import Trl
from .research_step import ResearchStep


class Service(MarketplaceDocument):
    pid = StringField()
    name = StringField()
    description = StringField()
    tagline = StringField()
    countries = ListField(StringField())
    order_type = StringField()
    rating = StringField()
    horizontal = BooleanField()
    categories = ListField(ReferenceField(Category))
    providers = ListField(ReferenceField(Provider))
    resource_organisation = ReferenceField(Provider)
    scientific_domains = ListField(ReferenceField(ScientificDomain))
    platforms = ListField(ReferenceField(Platform))
    target_users = ListField(ReferenceField(TargetUser))
    access_modes = ListField(ReferenceField(AccessMode))
    access_types = ListField(ReferenceField(AccessType))
    trls = ListField(ReferenceField(Trl))
    life_cycle_statuses = ListField(ReferenceField(LifeCycleStatus))
    research_steps = ListField(ReferenceField(ResearchStep))
    related_services = ListField(ReferenceField("Service"))
    required_services = ListField(ReferenceField("Service"))
    dataframe = BinaryField(blank=True)
    one_hot_tensor = ListField(IntField(), blank=True)
    dense_tensor = ListField(FloatField(), blank=True)
    status = StringField()

    meta = {
        "indexes": [
            {
                "fields": ["$name", "$description", "$tagline"],
                "default_language": "english",
                "weights": {"title": 10, "content": 5, "tagline": 2},
            }
        ]
    }
