"""User model"""

from flask_restx import fields
from app.api.v1.api import api

user = api.model(
    "User",
    {
        "id": fields.Integer(
            required=True,
            title="User ID",
            description="The unique identifier of a the user",
            example=1234,
        ),
        "scientific_domain_ids": fields.List(
            fields.String(
                title="Scientific domain ID",
                description="ID of the scientific domain",
                example=1234,
            ),
            required=True,
            title="Scientific domains IDs",
            description="List of scientific domains IDs",
        ),
        "category_ids": fields.List(
            fields.String(
                title="Category ID",
                description="ID of the category",
                example=1234,
            ),
            required=True,
            title="Categories IDs",
            description="List of categories IDs",
        ),
        "accessed_services_ids": fields.List(
            fields.String(
                title="Accessed service ID",
                description="ID of the accessed services",
                example=1234,
            ),
            required=True,
            title="Accessed services IDs",
            description="List of accessed services IDs",
        ),
    },
)
