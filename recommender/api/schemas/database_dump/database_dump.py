"""Database dump model"""

from flask_restx import fields

from recommender.api.schemas.common import api
from .access_mode import access_mode
from .access_type import access_type
from .category import category
from .life_cycle_status import life_cycle_status
from .platform import platform
from .project import project
from .provider import provider
from .scientific_domain import scientific_domain
from .service import service
from .target_user import target_user
from .trl import trl
from .user import user

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
        "projects": fields.List(
            fields.Nested(
                project,
                required=True,
                title="Project",
                description="Project  in the Marketplace Portal",
            ),
            required=True,
            title="Projects",
            description="List of projects",
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
