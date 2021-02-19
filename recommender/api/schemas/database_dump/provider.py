"""Provider model"""

from flask_restx import fields

from recommender.api.schemas.common import api

provider = api.model(
    "Provider",
    {
        "id": fields.Integer(
            required=True,
            title="Provider ID",
            description="The unique identifier of a the provider",
            example=1234,
        ),
        "name": fields.String(
            required=True,
            title="Name",
            description="The name of the provider",
            example="Cyfronet",
        ),
    },
)
