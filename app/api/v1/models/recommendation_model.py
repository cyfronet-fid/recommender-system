# pylint: disable=duplicate-code

"""Models for the recommendations endpoint"""

from flask_restx import fields, reqparse
from app.api.v1.api import api


recommendation_parser = reqparse.RequestParser()
recommendation_parser.add_argument(
    "location",
    required=True,
    type=str,
    help="An unambiguous identifier of the recommendation panel location on the page",
    default="services_list",
)

recommendation_parser.add_argument(
    "version",
    required=True,
    type=str,
    help="An unambiguous identifier of the version of the recommendation panel",
    default="v1",
)


recommendation_context = api.model(
    "Recommendation context",
    {
        "user_id": fields.Integer(
            required=True,
            title="User ID",
            description="The unique identifier of a user",
            example=1234,
        ),
        "recommendation_page_visit_id": fields.Integer(
            required=True,
            title="recommendation page visit ID",
            description="The unique identifier of the user presence on the specific \
            page in the specific time",
            example=1234,
        ),
        "search_phrase": fields.String(
            required=False,
            title="Search phrase",
            description="Search phrase text typed by user in the search panel  in \
            the context of this recommendation request",
            example="Cloud GPU",
        ),
        "filters": fields.List(
            fields.String(
                title="Filter",
                description="An unambiguous symbol of the filter",
                example="some_filter",
            ),
            required=False,
            title="Filters",
            description="A list of filters chosen by a user in the context of this \
            recommendation request",
        ),
    },
)

recommendation = fields.List(
    fields.Integer(
        title="Service ID", description="The unique identifier of a service"
    ),
    required=True,
    title="Recommended services list",
    description="List of the recommended services' IDs",
    example=[1234, 2345, 3456],
)
