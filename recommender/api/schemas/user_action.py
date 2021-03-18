"""Models for the user_actions endpoint"""

from flask_restx import fields

from .common import api

root = api.model(
    "Root info",
    {
        "type": fields.String(
            required=True,
            title="Root type",
            description="Informs whether user followed service from recommender box "
            "or just clicked service in the regular list in the services "
            "catalogue",
            example="recommendation_panel",
        ),
        "panel_id": fields.String(
            required=False,
            title="Root type",
            description="Field used only if the root type is recommendation_panel. "
            "The unique identifier of a recommender panel on the page",
            example="v1",
        ),
        "service_id": fields.Integer(
            required=False,
            title="Service ID",
            description="Field used only if the root type is recommendation_panel. "
            "The unique identifier of a recommended service clicked by "
            "the user",
            example=1,
        ),
    },
)

source = api.model(
    "Source",
    {
        "visit_id": fields.Integer(
            required=True,
            title="Visit ID",
            description="The unique identifier of a user presence on the user "
            "action's source page in the specific time",
            example=1,
        ),
        "page_id": fields.String(
            required=True,
            title="Page ID",
            description="The unique identifier of the user action's source page",
            example="services_catalogue_list",
        ),
        "root": fields.Nested(
            root,
            required=False,
            title="User journey root",
            description="If this is an action that starts in clicking service "
            "recommended in the recommendation panel or in the regular "
            "services list then it is a root action and this field should "
            "be populated",
        ),
    },
)

target = api.model(
    "Target",
    {
        "visit_id": fields.Integer(
            required=True,
            title="Visit ID",
            description="The unique identifier of a user presence on the user "
            "action's target page in the specific time",
            example=1,
        ),
        "page_id": fields.String(
            required=True,
            title="Page ID",
            description="The unique identifier of the user action's target page",
            example="service_about",
        ),
    },
)

action = api.model(
    "Action",
    {
        "type": fields.String(
            required=True,
            title="Type of the action",
            description="Type of the clicked element",
            example="button",
        ),
        "text": fields.String(
            required=True,
            title="Text on the clicked element",
            description="The unique identifier of the user action's target page",
            example="Details",
        ),
        "order": fields.Boolean(
            required=True,
            title="Order",
            description="Flag indicating whether action caused service ordering or "
            "not",
            example=True,
        ),
    },
)

user_action = api.model(
    "User Action",
    {
        "user_id": fields.Integer(
            required=False,
            title="User ID",
            description="The unique identifier of the logged user.",
            example=1234,
        ),
        "unique_id": fields.Integer(
            required=True,
            title="Not logged user ID",
            description="The unique identifier of the not logged user.",
            example=1234,
        ),
        "timestamp": fields.DateTime(
            dt_format="iso8601",
            required=True,
            title="Timestamp",
            description="The exact time of taking this action by the user in iso8601 "
            "format",
        ),
        "source": fields.Nested(source, required=True, title="User action source"),
        "target": fields.Nested(target, required=True, title="User action target"),
        "action": fields.Nested(
            action, required=True, title="User action", description="Action details"
        ),
    },
)
