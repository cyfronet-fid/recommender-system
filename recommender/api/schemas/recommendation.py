# pylint: disable=duplicate-code, wrong-import-position, fixme, unused-import

"""Models for the recommendations endpoint"""

from dotenv import load_dotenv

load_dotenv()

from flask_restx import fields
from recommender.utils import load_examples
from .common import api, NullableString, client_id_field

examples = load_examples()

search_data = api.model(
    "Search Data",
    {
        "q": fields.String(title="Search phrase", example=""),
        "categories": fields.List(
            fields.Integer(title="Category"), example=examples["categories"]
        ),
        "geographical_availabilities": fields.List(
            fields.String(title="Countries", example="WW"),
            example=examples["geographical_availabilities"],
        ),
        "order_type": fields.String(title="Order type", example="open_access"),
        "providers": fields.List(
            fields.Integer(title="Provider"), example=examples["providers"]
        ),
        "rating": fields.String(title="Rating", example="5"),
        "related_platforms": fields.List(
            fields.Integer(title="Related platforms"),
            example=examples["related_platforms"],
        ),
        "scientific_domains": fields.List(
            fields.Integer(title="Scientific domain"),
            example=examples["scientific_domains"],
        ),
        "sort": fields.String(title="Sort filter", example="_score"),
        "target_users": fields.List(
            fields.Integer(title="Target users"), example=examples["target_users"]
        ),
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
        "aai_uid": fields.String(
            required=False,
            title="AAI Token (UID)",
            description="The unique identifier of the logged "
            + "user in form of an AAI Token. ",
            example="64-characters@egi.eu",
        ),
        "unique_id": fields.String(
            required=True,
            title="Not logged user ID",
            description="The unique identifier of the not logged user.",
            example="5642c351-80fe-44cf-b606-304f2f338122",
        ),
        "client_id": client_id_field,
        "timestamp": fields.DateTime(
            dt_format="iso8601",
            required=True,
            title="Timestamp",
            description="The exact time of the recommendation request sending "
            "in iso8601 format",
        ),
        # TODO: there should be non-nullable string eventually.
        "visit_id": NullableString(
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
            required=False,
            title="Root type",
            description="The unique identifier of the recommender panel on the page",
            example="v1",
        ),
        "engine_version": fields.String(
            required=False,
            title="Engine version",
            description="Version of the recommendation engine",
            example="NCF",
        ),
        "candidates": fields.List(
            fields.Integer(
                title="Service ID", description="The unique identifier of the service"
            ),
            required=False,
            title="ElasticSearch services list",
            description="List of services IDs from ElasticSearch",
            example=[1, 2, 3, 4],
        ),
        "search_data": fields.Nested(search_data, required=True),
    },
)

recommendation = api.model(
    "Recommendations",
    {
        "panel_id": fields.String(
            required=True,
            title="Root type",
            description="The unique identifier of the recommender panel on the page",
            example="v1",
        ),
        "recommendations": fields.List(
            fields.Integer(
                title="Service ID", description="The unique identifier of the service"
            ),
            required=True,
            title="Recommended services list",
            description="List of the recommended services' IDs",
            example=[1234, 2345, 3456],
        ),
        "explanations": fields.List(
            fields.String(
                title="Explanation",
                description="Explanation of choice of the corresponding service",
            ),
            required=True,
            title="Explanations list",
            description="List of the recommended services explanations",
            example=[
                "some long explanation",
                "some long explanation",
                "some long explanation",
            ],
        ),
        "explanations_short": fields.List(
            fields.String(
                title="Short explanation",
                description="Short explanation of choice of the corresponding service",
            ),
            required=True,
            title="Short explanations list",
            description="List of the recommended services short explanations",
            example=[
                "some short explanation",
                "some short explanation",
                "some short explanation",
            ],
        ),
        "scores": fields.List(
            fields.Float(
                title="Service score",
                description="Score of the corresponding service on the basis"
                " of which the choice of recommendation has been made",
            ),
            required=True,
            title="Recommended services scores list",
            description="List of the recommended services scores",
            example=[0.7, 0.2, 0.1],
        ),
        "engine_version": fields.String(
            required=True,
            title="Engine version",
            description="Version of the recommendation engine",
            example="NCF",
        ),
    },
)
