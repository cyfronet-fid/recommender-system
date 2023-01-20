# pylint: disable=line-too-long
"""Provide service context for the UD"""

from typing import Dict, List, Type
from recommender.models import Service, MarketplaceDocument
from logger_config import get_logger

logger = get_logger(__name__)


def get_all_collection_ids(collection: Type[MarketplaceDocument]) -> List[int]:
    """Get all collection IDs"""
    return collection.objects.distinct("_id")


def service_ctx(body_request: Dict) -> Dict:
    """Provide service context as "all services" if candidates are not specified
    in the recommendations request's body"""
    if not body_request.get("candidates"):
        body_request["candidates"] = get_all_collection_ids(Service)

    return body_request
