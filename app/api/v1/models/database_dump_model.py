"""Models for the database dump endpoint"""

from flask_restx import fields
from app.api.v1.api import api

from app.api.v1.models.database_dump_submodels.access_mode_model import access_mode
from app.api.v1.models.database_dump_submodels.access_type_model import access_type
from app.api.v1.models.database_dump_submodels.category_model import category
from app.api.v1.models.database_dump_submodels.life_cycle_status_model import (
    life_cycle_status,
)
from app.api.v1.models.database_dump_submodels.platform_model import platform
from app.api.v1.models.database_dump_submodels.provider_model import provider
from app.api.v1.models.database_dump_submodels.scientific_domain_model import (
    scientific_domain,
)
from app.api.v1.models.database_dump_submodels.service_model import service
from app.api.v1.models.database_dump_submodels.target_user_model import target_user
from app.api.v1.models.database_dump_submodels.trl_model import trl
from app.api.v1.models.database_dump_submodels.user_model import user


database_dump = api.model(
    "Database dump",
    {
        "services": fields.List(
            fields.Nested(
                service,
                required=True,
                title="Service",
                description="Scientific Service",
            ),
            required=True,
            title="Scientific Services",
            description="List of scientific services",
        ),
        "users": fields.List(
            fields.Nested(
                user,
                required=True,
                title="User",
                description="User of the Marketplace Portal",
            ),
            required=True,
            title="Users",
            description="List of user",
        ),
        "categories": fields.List(
            fields.Nested(
                category,
                required=True,
                title="Category",
                description="Category",
            ),
            required=True,
            title="Categories",
            description="List of categories",
        ),
        "providers": fields.List(
            fields.Nested(
                provider,
                required=True,
                title="Provider",
                description="Provider",
            ),
            required=True,
            title="Providers",
            description="List of providers",
        ),
        "scientific_domains": fields.List(
            fields.Nested(
                scientific_domain,
                required=True,
                title="Scientific Domain",
                description="Scientific Domain",
            ),
            required=True,
            title="Scientific Domains",
            description="List of scientific domain",
        ),
        "platforms": fields.List(
            fields.Nested(
                platform,
                required=True,
                title="Platform",
                description="Platform",
            ),
            required=True,
            title="Platforms",
            description="List of platforms",
        ),
        "target_users": fields.List(
            fields.Nested(
                target_user,
                required=True,
                title="Target user",
                description="Target user",
            ),
            required=True,
            title="Target users",
            description="List of target users",
        ),
        "access_modes": fields.List(
            fields.Nested(
                access_mode,
                required=True,
                title="Access mode",
                description="Access mode",
            ),
            required=True,
            title="Access modes",
            description="List of access modes",
        ),
        "access_types": fields.List(
            fields.Nested(
                access_type,
                required=True,
                title="Access type",
                description="Access type",
            ),
            required=True,
            title="Access types",
            description="List of access types",
        ),
        "trls": fields.List(
            fields.Nested(
                trl,
                required=True,
                title="TRL",
                description="TRL",
            ),
            required=True,
            title="TRLs",
            description="List of TRLs",
        ),
        "life_cycle_statuses": fields.List(
            fields.Nested(
                life_cycle_status,
                required=True,
                title="TRL",
                description="TRL",
            ),
            required=True,
            title="TRLs",
            description="List of TRLs",
        ),
    },
)
