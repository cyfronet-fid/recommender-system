"""Category model"""

from flask_restx import fields

from recommender.api.schemas.common import api

catalogue = api.model(
    "Catalogue",
    {
        "id": fields.Integer(
            required=True,
            title="Catalogue ID",
            description="The unique identifier of a the catalogue",
            example=1234,
        ),
        "pid": fields.String(
            required=False,
            title="Pid",
            description="Pid of the catalogue",
            example="pid",
        ),
        "name": fields.String(
            required=True,
            title="Name",
            description="The name of the catalogue",
            example="EOSC",
        ),
    },
)
