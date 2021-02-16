# pylint: disable=missing-class-docstring

"""This module contains MongoDB models"""

from mongoengine import (
    Document,
    IntField,
    StringField,
    ListField,
    ReferenceField,
    DateTimeField,
    EmbeddedDocument,
    EmbeddedDocumentField,
    BooleanField,
)


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


class Recommendation(Document):
    logged_user = BooleanField()
    user = ReferenceField(User, blank=True)
    unique_id = IntField(blank=True)
    timestamp = DateTimeField()
    visit_id = IntField()
    page_id = StringField()
    panel_id = StringField()
    services = ListField(ReferenceField(Service))
    search_phrase = StringField()
    filters = ListField(StringField())


class Root(EmbeddedDocument):
    type = StringField()
    panel_id = StringField(blank=True)
    service = ReferenceField(Service)


class Target(EmbeddedDocument):
    meta = {"allow_inheritance": True}
    visit_id = IntField()
    page_id = StringField()


class Source(Target):
    root = EmbeddedDocumentField(Root, blank=True)


class Action(EmbeddedDocument):
    type = StringField()
    text = StringField()
    order = BooleanField()


class UserAction(Document):
    logged_user = BooleanField()
    user = ReferenceField(User, blank=True)
    unique_id = IntField(blank=True)
    timestamp = DateTimeField()
    source = EmbeddedDocumentField(Source)
    target = EmbeddedDocumentField(Target)
    action = EmbeddedDocumentField(Action)


class State(EmbeddedDocument):
    user = ReferenceField(User, blank=True)
    services_history = ListField(ReferenceField(Service))
    last_searchphrase = StringField()
    last_filters = ListField(StringField())


class Sars(Document):
    state = EmbeddedDocumentField(State)
    action = ListField(ReferenceField(Service))
    reward = ListField(ListField(StringField()))
    next_state = EmbeddedDocumentField(State)


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
