"""Category model"""

from flask_restx import fields

from recommender.api.schemas.common import api

category = api.model(
    "Category",
    {
        "id": fields.Integer(
            required=True,
            title="Category ID",
            description="The unique identifier of a the category",
            example=1234,
        ),
        "name": fields.String(
            required=True,
            title="Name",
            description="The name of the category",
            example="Networking",
        ),
    },
)
