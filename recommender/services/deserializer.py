# pylint: disable=no-member, line-too-long, logging-fstring-interpolation

"""Implementation of the user activity stub - for development purpose"""
import copy
from typing import List
from logger_config import get_logger

from recommender.models import (
    Recommendation,
    UserAction,
    Root,
    Source,
    Target,
    Action,
    SearchData,
    User,
)

logger = get_logger(__name__)


class Deserializer:
    """This class is responsible for deserialization.
    It converts API payload json dictionaries to MongoEngine database_dump"""

    @classmethod
    def deserialize_recommendation(cls, json_dict):
        """This method is used for deserialization of recommendation"""

        search_data_json_dict = json_dict.get("search_data", {})
        search_data = SearchData(
            q=search_data_json_dict.get("q"),
            categories=search_data_json_dict.get("categories"),
            geographical_availabilities=search_data_json_dict.get(
                "geographical_availabilities"
            ),
            order_type=search_data_json_dict.get("order_type"),
            providers=search_data_json_dict.get("providers"),
            related_platforms=search_data_json_dict.get("related_platforms"),
            scientific_domains=search_data_json_dict.get("scientific_domains"),
            sort=search_data_json_dict.get("sort"),
            target_users=search_data_json_dict.get("target_users"),
        ).save()

        recommendation = Recommendation(
            user=json_dict.get("user_id"),
            unique_id=json_dict.get("unique_id"),
            client_id=json_dict.get("client_id"),
            timestamp=json_dict.get("timestamp"),
            visit_id=json_dict.get("visit_id"),
            page_id=json_dict.get("page_id"),
            panel_id=json_dict.get("panel_id"),
            engine_version=json_dict.get("engine_version"),
            services=json_dict.get("services"),
            candidates=json_dict.get("candidates"),
            search_data=search_data,
        )

        return recommendation

    @classmethod
    def deserialize_user_action(cls, json_dict):
        """This method is used for deserialization of user_action"""

        source_data = json_dict.get("source", {})
        source1 = Source(
            visit_id=source_data.get("visit_id"), page_id=source_data.get("page_id")
        )

        root_data = json_dict.get("source", {}).get("root")
        if root_data is None:
            root = None
        else:
            root = Root(
                type=root_data.get("type"),
                panel_id=root_data.get("panel_id"),
                service=root_data.get("resource_id"),
            )
        source1.root = root

        target_data = json_dict.get("target", {})
        target = Target(**target_data)

        action_data = json_dict.get("action", {})
        action = Action(**action_data)

        user_id, aai_uid = json_dict.get("user_id"), json_dict.get("aai_uid")
        if aai_uid and not user_id:
            try:
                user_id = User.objects(aai_uid=aai_uid).first().id
            except AttributeError:
                user_id = None
                logger.warning(f"Could not find user with aai_uid: {aai_uid}")

        user_action = UserAction(
            user=user_id,
            unique_id=json_dict.get("unique_id"),
            timestamp=json_dict.get("timestamp"),
            client_id=json_dict.get("client_id"),
            source=source1,
            target=target,
            action=action,
        )

        return user_action


def deserialize_recommendation(body_request, services_ids: List, engine_name: str):
    """Enhance recommendation body request and save it to the DB"""
    rec_to_save = copy.deepcopy(body_request)
    rec_to_save["services"] = services_ids
    rec_to_save["engine_version"] = engine_name
    Deserializer.deserialize_recommendation(rec_to_save).save()
