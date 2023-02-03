"""Provider model"""

from flask_restx import fields

from recommender.api.schemas.common import api, NullableString

provider = api.model(
    "Provider",
    {
        "id": fields.Integer(
            required=True,
            title="Provider ID",
            description="The MP's unique ID of a provider",
            example=1234,
        ),
        "pid": NullableString(
            required=True,
            title="PID",
            description="The unique, global ID of a provider",
            example="string",
        ),
        "name": fields.String(
            required=True,
            title="Name",
            description="The name of the provider",
            example="Cyfronet",
        ),
    },
)
