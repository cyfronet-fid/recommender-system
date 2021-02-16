"""Life cycle status model"""

from flask_restx import fields
from app.api.v1.api import api

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
        "description": fields.String(
            title="Description",
            description="Description of the life cycle status",
            example="Some life cycle status description",
        ),
    },
)
