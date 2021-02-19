"""TRL model"""

from flask_restx import fields

from recommender.api.schemas.common import api, NullableString

trl = api.model(
    "TRL",
    {
        "id": fields.Integer(
            required=True,
            title="TRL",
            description="The unique identifier of a the TRL",
            example=1234,
        ),
        "name": fields.String(
            required=True,
            title="Name",
            description="The name of the TRL",
            example="some TRL name",
        ),
        "description": NullableString(
            required=True,
            title="Description",
            description="Description of the TRL",
            example="Some TRL description",
        ),
    },
)
