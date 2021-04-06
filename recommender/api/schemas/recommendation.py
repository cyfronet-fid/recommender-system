# pylint: disable=duplicate-code

"""Models for the recommendations endpoint"""

from flask_restx import fields

from .common import api


search_data = api.model(
    "Search Data",
    {
        "q": fields.String(title="Search phrase", example="Cloud GPU"),
        "category_id": fields.Integer(title="Category", example=1),
        "geographical_availabilities": fields.List(
            fields.String(title="Countries", example="PL")
        ),
        "order_type": fields.String(title="Order type", example="open_access"),
        "providers": fields.List(fields.Integer(title="Provider", example=1)),
        "rating": fields.String(title="Rating", example="5"),
        "related_platforms": fields.List(
            fields.Integer(title="Related platforms", example=1)
        ),
        "scientific_domains": fields.List(
            fields.Integer(title="Scientific domain", example=1)
        ),
        "sort": fields.String(title="Sort filter", example="_score"),
        "target_users": fields.List(fields.Integer(title="Target users", example=1)),
    },
)

recommendation_context = api.model(
    "Recommendation context",
    {
        "user_id": fields.Integer(
            required=False,
            title="User ID",
            description="The unique identifier of the logged user. ",
            example=1,
        ),
        "unique_id": fields.String(
            required=True,
            title="Not logged user ID",
            description="The unique identifier of the not logged user.",
            example="5642c351-80fe-44cf-b606-304f2f338122",
        ),
        "timestamp": fields.DateTime(
            dt_format="iso8601",
            required=True,
            title="Timestamp",
            description="The exact time of the recommendation request sending "
            "in iso8601 format",
        ),
        "visit_id": fields.String(
            required=True,
            title="recommendation page visit ID",
            description="The unique identifier of the user presence on the "
            "recommendation page in the specific time (could be "
            "a function of above fields)",
            example="202090a4-de4c-4230-acba-6e2931d9e37c",
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
            example="v1",
        ),
        "search_data": fields.Nested(search_data, required=True),
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
