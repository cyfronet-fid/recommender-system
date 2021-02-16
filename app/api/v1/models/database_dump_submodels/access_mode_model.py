"""Access mode model"""

from flask_restx import fields
from app.api.v1.api import api

access_mode = api.model(
    "Access mode",
    {
        "id": fields.Integer(
            required=True,
            title="Access mode",
            description="The unique identifier of a the access mode",
            example=1234,
        ),
        "name": fields.String(
            required=True,
            title="Name",
            description="The name of the access mode",
            example="open access",
        ),
        "description": fields.String(
            title="Description",
            description="Description of the access mode",
            example="The access is open for everyone",
        ),
    },
)
