# pylint: disable=line-too-long
"""Provide service context for the UD"""

from typing import Dict, List, Type
from recommender.models import Service, MarketplaceDocument
from recommender.errors import ServicesContextNotProvidedError
from logger_config import get_logger

logger = get_logger(__name__)


def get_all_collection_ids(collection: Type[MarketplaceDocument]) -> List[int]:
    """Get all collection IDs"""
    return collection.objects.distinct("_id")


def service_ctx(body_request: Dict) -> Dict:
    """Provide service context as "all services" if candidates are not specified
    in the recommendations request's body, for the purpose of the User Dashboard"""

    candidates = body_request.get("candidates")
    page_id = body_request.get("page_id")

    if candidates is None:
        if page_id == "/dashboard":  # case for the UD
            body_request["candidates"] = get_all_collection_ids(Service)
        else:
            logger.error(
                "Candidates not provided. Only for the context of page_id == '/dashboard' they are NOT required"
            )
            raise ServicesContextNotProvidedError()

    return body_request
