from pymodm import fields, MongoModel


class Category(MongoModel):
    id = fields.IntegerField(primary_key=True)
    name = fields.CharField()


class Provider(MongoModel):
    id = fields.IntegerField(primary_key=True)
    name = fields.CharField()


class ScientificDomain(MongoModel):
    id = fields.IntegerField(primary_key=True)
    name = fields.CharField()


class Platform(MongoModel):
    id = fields.IntegerField(primary_key=True)
    name = fields.CharField()


class TargetUser(MongoModel):
    id = fields.IntegerField(primary_key=True)
    name = fields.CharField()
    description = fields.CharField()


class AccessMode(MongoModel):
    id = fields.IntegerField(primary_key=True)
    name = fields.CharField()
    description = fields.CharField()


class AccessType(MongoModel):
    id = fields.IntegerField(primary_key=True)
    name = fields.CharField()
    description = fields.CharField()


class Trl(MongoModel):
    id = fields.IntegerField(primary_key=True)
    name = fields.CharField()
    description = fields.CharField()


class LifeCycleStatus(MongoModel):
    id = fields.IntegerField(primary_key=True)
    name = fields.CharField()
    description = fields.CharField()


class Service(MongoModel):
    id = fields.IntegerField(primary_key=True)
    name = fields.CharField()
    description = fields.CharField()
    tagline = fields.CharField()
    countries = fields.ListField(fields.CharField())
    categories = fields.ListField(fields.ReferenceField("Category"))
    providers = fields.ListField(fields.ReferenceField("Provider"))
    resource_organisation = fields.ReferenceField("Provider")
    scientific_domains = fields.ListField(fields.ReferenceField("ScientificDomain"))
    platforms = fields.ListField(fields.ReferenceField("Platform"))
    target_users = fields.ListField(fields.ReferenceField("TargetUser"))
    access_modes = fields.ListField(fields.ReferenceField("AccessMode"))
    access_types = fields.ListField(fields.ReferenceField("AccessType"))
    trls = fields.ListField(fields.ReferenceField("Trl"))
    life_cycle_statuses = fields.ListField(fields.ReferenceField(LifeCycleStatus))
    # Integer field instead of ReferenceField to avoid circular references
    related_services = fields.ListField(fields.IntegerField())
    # Integer field instead of ReferenceField to avoid circular references
    required_services = fields.ListField(fields.IntegerField())


class User(MongoModel):
    id = fields.IntegerField(primary_key=True)
    scientific_domains = fields.ListField(fields.ReferenceField("ScientificDomain"))
    categories = fields.ListField(fields.ReferenceField("Category"))
    accessed_services = fields.ListField(fields.ReferenceField("Service"))
