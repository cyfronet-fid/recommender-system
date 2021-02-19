"""Scientific domain model"""

from flask_restx import fields

from recommender.api.schemas.common import api

scientific_domain = api.model(
    "Scientific domain",
    {
        "id": fields.Integer(
            required=True,
            title="Scientific domain ID",
            description="The unique identifier of a the scientific domain",
            example=1234,
        ),
        "name": fields.String(
            required=True,
            title="Name",
            description="The name of the scientific domain",
            example="Natural Sciences",
        ),
    },
)
