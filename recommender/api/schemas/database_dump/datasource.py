"""Datasource model"""

from flask_restx import fields

from recommender.api.schemas.common import api

datasource = api.model(
    "Datasource",
    {
        "id": fields.Integer(
            required=True,
            title="Datasource ID",
            example=1234,
        ),
        "pid": fields.String(
            required=False,
            title="Pid",
            description="Pid",
            example="pid",
        ),
        "name": fields.String(
            required=True,
            title="Name",
            example="Cloud GPU",
        ),
        "description": fields.String(
            required=True,
            title="Description",
            example="Service providing GPU cluster on demand",
        ),
        "tagline": fields.String(
            required=True,
            title="Tagline",
            example="State-of-the-art service",
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
        "order_type": fields.String(
            required=True,
            title="Order type",
            description="Order type",
            example="Order required",
        ),
        "horizontal": fields.Boolean(required=False, title="Horizontal"),
        "standards": fields.List(
            fields.String(title="Standard"), required=False, title="Standards"
        ),
        "open_source_technologies": fields.List(
            fields.String(title="Open source technology"),
            title="Open source technologies",
            required=False,
        ),
        "categories": fields.List(
            fields.Integer(
                title="Category ID",
                description="ID of the category",
                example=1234,
            ),
            required=True,
            title="Categories IDs",
            description="List of categories IDs",
        ),
        "catalogues": fields.List(
            fields.Integer(
                title="Catalogue ID",
                description="ID of the catalogue",
                example=1234,
            ),
            required=False,
            title="Catalogue IDs",
            description="List of catalogue IDs",
        ),
        "providers": fields.List(
            fields.Integer(
                title="Provider ID",
                description="ID of the provider",
                example=1234,
            ),
            required=True,
            title="Providers IDs",
            description="List of providers IDs",
        ),
        "resource_organisation": fields.Integer(
            title="Resource organisation ID",
            required=True,
            description="Resource organisations ID",
            example=1234,
        ),
        "scientific_domains": fields.List(
            fields.Integer(
                title="Scientific domain ID",
                description="ID of the scientific domain",
                example=1234,
            ),
            required=True,
            title="Scientific domains IDs",
            description="List of scientific domains IDs",
        ),
        "platforms": fields.List(
            fields.Integer(
                title="Platform ID",
                example=1234,
            ),
            required=True,
            title="Platforms IDs",
        ),
        "target_users": fields.List(
            fields.Integer(
                title="Target user ID",
                example=1234,
            ),
            required=True,
            title="Target users IDs",
        ),
        "related_services": fields.List(
            fields.Integer(
                title="Related service ID",
                example=1234,
            ),
            required=True,
            title="Related services IDs",
        ),
        "access_modes": fields.List(
            fields.Integer(
                title="Access mode ID",
                example=1234,
            ),
            required=True,
            title="Access modes IDs",
        ),
        "access_types": fields.List(
            fields.Integer(
                title="Access type ID",
                example=1234,
            ),
            required=True,
            title="Access types IDs",
        ),
        "trls": fields.List(
            fields.Integer(
                title="TRL ID",
                example=1234,
            ),
            required=True,
            title="TRLs IDs",
        ),
        "life_cycle_statuses": fields.List(
            fields.Integer(
                title="Life cycle status ID",
                example=1234,
            ),
            required=True,
            title="Life cycle statuses IDs",
        ),
    },
)
