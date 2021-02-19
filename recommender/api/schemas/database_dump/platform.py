"""Platform model"""

from flask_restx import fields

from recommender.api.schemas.common import api

platform = api.model(
    "Platform",
    {
        "id": fields.Integer(
            required=True,
            title="Platform ID",
            description="The unique identifier of a the platform",
            example=1234,
        ),
        "name": fields.String(
            required=True,
            title="Name",
            description="The name of the platform",
            example="Digital Cloud",
        ),
    },
)
