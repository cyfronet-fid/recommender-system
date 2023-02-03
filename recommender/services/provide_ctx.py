# pylint: disable=line-too-long
"""Provide service context for the UD"""

from typing import Dict, List, Type
from recommender.models import Service, MarketplaceDocument
from logger_config import get_logger

logger = get_logger(__name__)


def provide_ctx(body_request: Dict) -> Dict:
    """Provide request context"""
    body_request = panel_id_ctx(body_request)
    body_request = service_ctx(body_request)
    return body_request


def panel_id_ctx(body_request: Dict) -> Dict:
    """Provide panel_id context"""
    panel_id = body_request.get("panel_id")
    if not panel_id or panel_id == "service":
        body_request["panel_id"] = "v1"

    return body_request


def service_ctx(body_request: Dict) -> Dict:
    """Provide service context as "all services" if candidates are not specified
    in the recommendations request's body"""
    if not body_request.get("candidates"):
        body_request["candidates"] = get_all_collection_ids(Service)

    return body_request


def get_all_collection_ids(collection: Type[MarketplaceDocument]) -> List[int]:
    """Get all collection IDs"""
    return collection.objects.distinct("_id")
