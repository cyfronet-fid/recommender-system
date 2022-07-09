"""Project model"""

from flask_restx import fields

from recommender.api.schemas.common import api

project = api.model(
    "Project",
    {
        "id": fields.Integer(
            required=True,
            title="Project ID",
            description="The unique identifier of a the project",
            example=1234,
        ),
        "user_id": fields.Integer(
            required=True,
            title="User ID",
            description="ID of the owner of the project",
            example=1234,
        ),
        "services": fields.List(
            fields.Integer(
                title="Included service ID",
                description="ID of the included services",
                example=1234,
            ),
            required=True,
            title="Included services IDs",
            description="List of included services IDs",
        ),
    },
)
