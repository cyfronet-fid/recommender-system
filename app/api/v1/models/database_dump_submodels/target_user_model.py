"""Target user model"""

from flask_restx import fields
from app.api.v1.api import api

target_user = api.model(
    "Target user",
    {
        "id": fields.Integer(
            required=True,
            title="Target user",
            description="The unique identifier of a the target user",
            example=1234,
        ),
        "name": fields.String(
            required=True,
            title="Name",
            description="The name of the target user",
            example="Business",
        ),
        "description": fields.String(
            required=True,
            title="Description",
            description="Description of the target user",
            example="The user of the service is a company",
        ),
    },
)
