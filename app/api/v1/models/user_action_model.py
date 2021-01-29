"""Models for the user_actions endpoint"""

from flask_restx import fields
from app.api.v1.api import api


root = api.model(
    "Root info",
    {
        "root_type": fields.String(
            required=True,
            title="Root Type",
            description="Field informing whether user followed recommended service or \
            clicked just service in the regular list or in some other place",
            example="recommendation_panel",
        ),
        "location": fields.String(
            required=False,
            title="Recommendation panel location",
            description="Field used only if the root type is recommendation_panel. \
            On one page there can be few recommendation panels so this field allows \
            to distinguish them",
            example="bottom_left_recommendation panel",
        ),
        "version": fields.String(
            required=False,
            title="Recommendation panel version",
            description="Field used only if the root type is recommendation_panel. \
            There can be few versions of the same recommendation panel with e.g. \
            different number of recommended services",
            example="v1",
        ),
        "service_id": fields.Integer(
            required=False,
            title="Service ID",
            description="Field used only if the root type is recommendation_panel. \
            The unique identifier of a recommended service clicked by the user",
            example=1234,
        ),
    },
)

user_action = api.model(
    "User Action",
    {
        "user_id": fields.Integer(
            required=True,
            title="User ID",
            description="The unique identifier of a user",
            example=1234,
        ),
        "source_page_visit_id": fields.Integer(
            required=True,
            title="Source page visit ID",
            description="The unique identifier of a user presence on the user action \
            source page in the specific time",
            example=1234,
        ),
        "target_page_visit_id": fields.Integer(
            required=True,
            title="Target page visit ID",
            description="The unique identifier of a user presence on the user action \
            target page in the specific time",
            example=1234,
        ),
        "user_action": fields.String(
            required=True,
            title="User action",
            description="An unambiguous symbol of the user action",
            example="Regular link click",
        ),
        "root": fields.Nested(
            root,
            required=True,
            title="User journey root",
            description="If this is an action that starts in clicking service \
            recommended in the recommendation panel or in the regular services list \
            then it is a root action and this field should be populated",
        ),
    },
)
