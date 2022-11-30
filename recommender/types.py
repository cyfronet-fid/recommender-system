"""Pydantic types, in the future this might be changed into package"""
import datetime
import uuid
from typing import Optional

# pylint: disable=no-name-in-module
from pydantic import BaseModel, Field


# pylint: disable=too-few-public-methods, missing-class-docstring
class UserJourneyRoot(BaseModel):
    type: str = Field(
        title="Root type",
        description="Informs whether user followed service from recommender box "
        "or just clicked service in the regular list in the services "
        "catalogue",
        example="recommendation_panel",
    )
    panel_id: Optional[str] = Field(
        default=None,
        title="Root type",
        description="Field used only if the root type is recommendation_panel. "
        "The unique identifier of a recommender panel on the page",
        example="v1",
    )
    resource_type: Optional[str] = Field(
        default=None,
        title="Resource type",
        description="The type of the resource clicked by the user. Currently"
        " only supported type is a `service`",
        example="service",
    )
    resource_id: Optional[int] = Field(
        default=None,
        title="Resource ID",
        description="The unique identifier of a recommended resource clicked"
        " by the user. Currently it can be only"
        " id of the service.",
        example=1,
    )


# pylint: disable=too-few-public-methods, missing-class-docstring
class UserActionSource(BaseModel):
    visit_id: Optional[uuid.UUID] = Field(
        default=None,
        title="Visit ID",
        description="The unique identifier of a user presence on the user "
        "action's source page in the specific time",
        example="202090a4-de4c-4230-acba-6e2931d9e37c",
    )
    page_id: str = Field(
        title="Page ID",
        description="The unique identifier of the user action's source page",
        example="services_catalogue_list",
    )
    root: Optional[UserJourneyRoot] = Field(
        default=None,
        title="User journey root",
        description="If this is an action that starts in clicking service "
        "recommended in the recommendation panel or in the regular "
        "services list then it is a root action and this field should "
        "be populated",
    )


# pylint: disable=too-few-public-methods, missing-class-docstring
class UserActionTarget(BaseModel):
    visit_id: uuid.UUID = Field(
        required=True,
        title="Visit ID",
        description="The unique identifier of a user presence on the user "
        "action's target page in the specific time",
        example="9f543b80-dd5b-409b-a619-6312a0b04f4f",
    )

    page_id: str = Field(
        required=True,
        title="Page ID",
        description="The unique identifier of the user action's target page",
        example="service_about",
    )


# pylint: disable=too-few-public-methods, missing-class-docstring
class Action(BaseModel):
    type: str = Field(
        title="Type of the action",
        description="Type of the clicked element",
        example="button",
    )
    text: str = Field(
        title="Text on the clicked element",
        description="The unique identifier of the user action's target page",
        example="Details",
    )
    order: bool = (
        Field(
            title="Order",
            description="Flag indicating whether action caused service ordering or not",
        ),
    )


# pylint: disable=too-few-public-methods, missing-class-docstring
class UserAction(BaseModel):
    user_id: Optional[int] = Field(
        title="User ID",
        description="The unique identifier of the logged user.",
        example=1234,
    )
    unique_id: uuid.UUID = Field(
        title="Not logged user ID",
        description="The unique identifier of the not logged user.",
        example="5642c351-80fe-44cf-b606-304f2f338122",
    )
    timestamp: datetime.datetime = Field(
        title="Timestamp",
        description="The exact time of taking this action by the user",
    )
    source: UserActionSource
    target: UserActionTarget
    action: Action
