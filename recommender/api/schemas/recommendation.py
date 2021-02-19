# pylint: disable=duplicate-code

"""Models for the recommendations endpoint"""

from flask_restx import fields

from .common import api

recommendation_context = api.model(
    "Recommendation context",
    {
        "logged_user": fields.Boolean(
            required=True,
            title="Logged user",
            description="Flag indicating whether user is logged or not",
            example=True,
        ),
        "user_id": fields.Integer(
            required=False,
            title="User ID",
            description="The unique identifier of the logged user. "
            "Specified only if logged_user=True",
            example=1234,
        ),
        "unique_id": fields.Integer(
            required=False,
            title="Not logged user ID",
            description="The unique identifier of the not logged user. "
            "Specified only if logged_user=False",
            example=1234,
        ),
        "timestamp": fields.DateTime(
            dt_format="iso8601",
            required=True,
            title="Timestamp",
            description="The exact time of the recommendation request sending "
            "in iso8601 format",
        ),
        "visit_id": fields.Integer(
            required=True,
            title="recommendation page visit ID",
            description="The unique identifier of the user presence on the "
            "recommendation page in the specific time (could be "
            "a function of above fields)",
            example=1234,
        ),
        "page_id": fields.String(
            required=True,
            title="Page ID",
            description="The unique identifier of the page with recommendation panel",
            example="some_page_identifier",
        ),
        "panel_id": fields.String(
            required=True,
            title="Root type",
            description="The unique identifier of the recommender panel on the page",
            example="version_A",
        ),
        "search_phrase": fields.String(
            required=True,
            title="Search phrase",
            description="Search phrase text typed by user in the search panel  in "
            "the context of this recommendation request",
            example="Cloud GPU",
        ),
        "filters": fields.List(
            fields.String(
                title="Filter",
                description="An unambiguous symbol of the filter",
                example="some_filter",
            ),
            required=True,
            title="Filters",
            description="A list of filters chosen by a user in the context of this "
            "recommendation request",
        ),
    },
)

recommendation = api.model(
    "Recommendations",
    {
        "recommendations": fields.List(
            fields.Integer(
                title="Service ID", description="The unique identifier of the service"
            ),
            required=True,
            title="Recommended services list",
            description="List of the recommended services' IDs",
            example=[1234, 2345, 3456],
        )
    },
)
