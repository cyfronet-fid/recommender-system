"""Models for the database dump endpoint"""

from flask_restplus import fields
from app.api.v1.api import api

row = api.model(
    "Row",
    {
        "id": fields.Integer(
            required=True,
            title="Row ID",
            description="The unique identifier of the record",
            example=1234,
        ),
    },
)

table = api.model(
    "Table",
    {
        "table": fields.String(
            required=True,
            title="Table name",
            description="The name of the table",
            example="services",
        ),
        "rows": fields.List(
            fields.Nested(
                row,
                required=True,
                title="Row",
                description="Table row",
            ),
            required=True,
            title="Table rows",
            description="Records of the table",
        ),
    },
)

database_dump = api.model(
    "Database dump",
    {
        "tables": fields.List(
            fields.Nested(
                table,
                required=True,
                title="Table",
                description="Table of the dumped database",
            ),
            required=True,
            title="Tables",
            description="Tables of the dumped database",
        )
    },
)
