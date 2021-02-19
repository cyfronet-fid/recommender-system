"""Access type"""

from flask_restx import fields

from recommender.api.schemas.common import api, NullableString

access_type = api.model(
    "Access type",
    {
        "id": fields.Integer(
            required=True,
            title="Access type",
            description="The unique identifier of a the access type",
            example=1234,
        ),
        "name": fields.String(
            required=True,
            title="Name",
            description="The name of the access type",
            example="remote access",
        ),
        "description": NullableString(
            required=True,
            title="Description",
            description="Description of the access type",
            example="Access can be remote",
        ),
    },
)
