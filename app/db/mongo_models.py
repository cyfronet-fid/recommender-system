# pylint: disable=missing-class-docstring

"""Mongo model definitions"""
from mongoengine import Document, IntField, StringField, ListField, ReferenceField


class BaseDocument(Document):
    id = IntField(primary_key=True)
    meta = {
        "allow_inheritance": True,
        "abstract": True,
    }


class Category(BaseDocument):
    name = StringField()


class Provider(BaseDocument):
    name = StringField()


class ScientificDomain(BaseDocument):
    name = StringField()


class Platform(BaseDocument):
    name = StringField()


class TargetUser(BaseDocument):
    name = StringField()
    description = StringField()


class AccessMode(BaseDocument):
    name = StringField()
    description = StringField()


class AccessType(BaseDocument):
    name = StringField()
    description = StringField()


class Trl(BaseDocument):
    name = StringField()
    description = StringField()


class LifeCycleStatus(BaseDocument):
    name = StringField()
    description = StringField()


class Service(BaseDocument):
    name = StringField()
    description = StringField()
    tagline = StringField()
    countries = ListField(StringField())
    categories = ListField(ReferenceField("Category"))
    providers = ListField(ReferenceField("Provider"))
    resource_organisation = ReferenceField("Provider")
    scientific_domains = ListField(ReferenceField("ScientificDomain"))
    platforms = ListField(ReferenceField("Platform"))
    target_users = ListField(ReferenceField("TargetUser"))
    access_modes = ListField(ReferenceField("AccessMode"))
    access_types = ListField(ReferenceField("AccessType"))
    trls = ListField(ReferenceField("Trl"))
    life_cycle_statuses = ListField(ReferenceField(LifeCycleStatus))
    related_services = ListField(ReferenceField("Service"))
    required_services = ListField(ReferenceField("Service"))


class User(BaseDocument):
    scientific_domains = ListField(ReferenceField("ScientificDomain"))
    categories = ListField(ReferenceField("Category"))
    accessed_services = ListField(ReferenceField("Service"))


MP_DUMP_MODEL_CLASSES = [
    Category,
    Provider,
    ScientificDomain,
    Platform,
    TargetUser,
    AccessMode,
    AccessType,
    Trl,
    LifeCycleStatus,
    User,
    Service,
]
