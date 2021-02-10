"""Service model"""

from flask_restx import fields
from app.api.v1.api import api

service = api.model(
    "Service",
    {
        "id": fields.Integer(
            required=True,
            title="Service ID",
            description="The unique identifier of a the service",
            example=1234,
        ),
        "name": fields.String(
            required=True,
            title="Name",
            description="Name of the service",
            example="Cloud GPU",
        ),
        "description": fields.String(
            required=True,
            title="Description",
            description="description of the service",
            example="Service providing GPU cluster on demand",
        ),
        "tagline": fields.List(
            fields.String(
                title="Tag",
                description="Tag of the service",
                example="gpu",
            ),
            required=True,
            title="Tag line",
            description="A list of service's tags",
        ),
        "countries": fields.List(
            fields.String(
                title="Country",
                description="Country Alpha-2 code",
                example="PL",
            ),
            required=True,
            title="Countries",
            description="List of countries",
        ),
        "categories": fields.List(
            fields.String(
                title="Category ID",
                description="ID of the category",
                example=1234,
            ),
            required=True,
            title="Categories IDs",
            description="List of categories IDs",
        ),
        "providers": fields.List(
            fields.String(
                title="Provider ID",
                description="ID of the provider",
                example=1234,
            ),
            required=True,
            title="Providers IDs",
            description="List of providers IDs",
        ),
        "resource_organisation": fields.List(
            fields.String(
                title="Resource organisation ID",
                description="ID of the resource organisation",
                example=1234,
            ),
            required=True,
            title="Resource organisations IDs",
            description="List of resource organisations IDs",
        ),
        "scientific_domains": fields.List(
            fields.String(
                title="Scientific domain ID",
                description="ID of the scientific domain",
                example=1234,
            ),
            required=True,
            title="Scientific domains IDs",
            description="List of scientific domains IDs",
        ),
        "platforms": fields.List(
            fields.String(
                title="Platform ID",
                description="ID of the service's platform",
                example=1234,
            ),
            required=True,
            title="Platforms IDs",
            description="List of service's platforms IDs",
        ),
        "target_users": fields.List(
            fields.String(
                title="Target user ID",
                description="ID of the service's target user",
                example=1234,
            ),
            required=True,
            title="Target users IDs",
            description="List of service's target users IDs",
        ),
        "related_services": fields.List(
            fields.String(
                title="Related service ID",
                description="ID of the service's related service",
                example=1234,
            ),
            required=True,
            title="Related services IDs",
            description="List of service's related services IDs",
        ),
        "access_modes": fields.List(
            fields.String(
                title="Access mode ID",
                description="ID of the service's access mode",
                example=1234,
            ),
            required=True,
            title="Access modes IDs",
            description="List of service's access modes IDs",
        ),
        "access_types": fields.List(
            fields.String(
                title="Access type ID",
                description="ID of the service's access type",
                example=1234,
            ),
            required=True,
            title="Access types IDs",
            description="List of service's access types IDs",
        ),
        "trls": fields.List(
            fields.String(
                title="TRL ID",
                description="ID of the service's Technology readiness level",
                example=1234,
            ),
            required=True,
            title="TRLs IDs",
            description="List of service's TRLs IDs",
        ),
        "life_cycle_statuses": fields.List(
            fields.String(
                title="Life cycle status ID",
                description="ID of the service's life cycle status",
                example=1234,
            ),
            required=True,
            title="Life cycle statuses IDs",
            description="List of service's life cycle statuses IDs",
        ),
    },
)
