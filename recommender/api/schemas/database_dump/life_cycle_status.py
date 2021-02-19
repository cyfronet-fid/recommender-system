"""Life cycle status model"""

from flask_restx import fields

from recommender.api.schemas.common import api, NullableString

life_cycle_status = api.model(
    "Life cycle status",
    {
        "id": fields.Integer(
            required=True,
            title="Life cycle status",
            description="The unique identifier of a the life cycle status",
            example=1234,
        ),
        "name": fields.String(
            required=True,
            title="Name",
            description="The name of the life cycle status",
            example="some TRL name",
        ),
        "description": NullableString(
            required=True,
            title="Description",
            description="Description of the life cycle status",
            example="Some life cycle status description",
        ),
    },
)
