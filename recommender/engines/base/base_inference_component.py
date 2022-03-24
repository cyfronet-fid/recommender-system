# pylint: disable=invalid-name, no-self-use, no-member, too-few-public-methods, fixme

"""Implementation of the base agent recommmender. Every other agent should
 inherit from this one."""

from abc import ABC, abstractmethod
import random
from typing import Dict, Any, List, Tuple

from recommender.engines.panel_id_to_services_number_mapping import PANEL_ID_TO_K
from recommender.errors import (
    InsufficientRecommendationSpaceError,
    InvalidRecommendationPanelIDError,
)
from recommender.models import User, SearchData
from recommender.services.fts import retrieve_services_for_recommendation
from logger_config import get_logger

logger = get_logger(__name__)


class BaseInferenceComponent(ABC):
    """
    Base Recommender class with basic functionality
    """

    def __init__(self, K: int) -> None:
        """
        Initialization function.
        """
        self.K = K
        if self.K not in PANEL_ID_TO_K.values():
            raise InvalidRecommendationPanelIDError()
        self._load_models()

    def __call__(self, context: Dict[str, Any]) -> List[int]:
        """
        This function allows for getting recommended services for the
        recommendation endpoint based on recommendation context.

        Args:
            context: json dict from the /recommendations endpoint request.

        Returns:
            Tuple of recommended services ids.
        """

        user = self._get_user(context)
        elastic_services, search_data = self._get_elastic_services_and_search_data(
            context
        )

        if user is not None:
            return self._for_logged_user(user, elastic_services, search_data)
        return self._for_anonymous_user(elastic_services)

    @abstractmethod
    def _load_models(self) -> None:
        """
        It loads model or models needed for recommendations and raise
         exception if it is not available in the database.
        """

    def _get_user(self, context: Dict[str, Any]) -> User:
        """
        Get the user from the context.

        Args:
            context: context json  from the /recommendations endpoint request.

        Returns:
            User.
        """

        user = None
        if context.get("user_id") is not None:
            user = User.objects(id=context.get("user_id")).first()

        return user

    @staticmethod
    def _get_elastic_services_and_search_data(
        context: Dict[str, Any]
    ) -> [Tuple[int], SearchData]:
        return tuple(context.get("elastic_services")), context.get("search_data")

    @abstractmethod
    def _for_logged_user(
        self, user: User, elastic_services: Tuple[int], search_data: SearchData
    ) -> List[int]:
        """
        Generate recommendation for logged user

        Args:
            user: user for whom recommendation will be generated.
            elastic_services: item space from the Marketplace.
            search_data: search phrase and filters information for narrowing
             down an item space.

        Returns:
            recommended_services_ids: List of recommended services ids.
        """

    def _for_anonymous_user(self, elastic_services: Tuple[int]) -> List[int]:
        """
        Generate recommendation for anonymous user

        Args:
            elastic_services: item space from the Marketplace.

        Returns:
            recommended_services_ids: List of recommended services ids.
        """

        candidate_services = list(
            retrieve_services_for_recommendation(elastic_services)
        )
        if len(candidate_services) < self.K:
            raise InsufficientRecommendationSpaceError()
        recommended_services = random.sample(list(candidate_services), self.K)
        recommended_services_ids = [s.id for s in recommended_services]

        return recommended_services_ids
