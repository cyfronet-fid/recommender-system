"""Life cycle status model"""

from flask_restx import fields

from recommender.api.schemas.common import api, NullableString

research_step = api.model(
    "Research step",
    {
        "id": fields.Integer(
            required=True,
            title="Research step",
            description="The unique identifier of a the research step",
            example=1234,
        ),
        "name": fields.String(
            required=True,
            title="Name",
            description="The name of the research step",
            example="some research step name",
        ),
        "description": NullableString(
            required=True,
            title="Description",
            description="Description of the research step",
            example="Some research step description",
        ),
    },
)
