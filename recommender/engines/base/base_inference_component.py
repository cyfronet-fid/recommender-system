# pylint: disable=invalid-name, no-member, too-few-public-methods, line-too-long

"""Implementation of the base recommender engine"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple

from recommender.engines.panel_id_to_services_number_mapping import PANEL_ID_TO_K
from recommender.errors import (
    InvalidRecommendationPanelIDError,
    InsufficientRecommendationSpaceError,
    UserCannotBeIdentified,
)
from recommender.models import User, SearchData


class BaseInferenceComponent(ABC):
    """
    Base Recommender class with basic functionality.
    Used in all available recommender engines.
    """

    engine_name = None

    def __init__(self, K: int) -> None:
        self.K = K
        if self.K not in PANEL_ID_TO_K.values():
            raise InvalidRecommendationPanelIDError()

    @abstractmethod
    def __call__(self, context: Dict[str, Any]) -> List[int]:
        """
        Get recommendations.

        Args:
            context: json dict from the /recommendations endpoint request.

        Returns:
            List of recommended services ids.
        """

    def _get_recommendation_context(
        self, context: Dict[str, Any]
    ) -> [Tuple[int], SearchData]:
        elastic_services = tuple(context.get("elastic_services") or ())
        search_data = context.get("search_data")

        if len(elastic_services) < self.K:
            raise InsufficientRecommendationSpaceError()

        return elastic_services, search_data


class MLEngineInferenceComponent(BaseInferenceComponent):
    """
    Recommender class for machine learning engines that serves recommendations to logged-in users.
    Used in NCF and RL engines.
    """

    def __init__(self, K: int) -> None:
        super().__init__(K)
        self._load_models()

    def __call__(self, context: Dict[str, Any]) -> List[int]:
        """
        Get recommendations for logged-in user.

        Args:
            context: json dict from the /recommendations endpoint request.

        Returns:
            List of recommended services ids.
        """

        user = self._get_user(context)
        elastic_services, search_data = self._get_recommendation_context(context)

        return self._generate_recommendations(user, elastic_services, search_data)

    @abstractmethod
    def _load_models(self) -> None:
        """
        It loads model or models needed for recommendations and raise
         exception if it is not available in the database.
        """

    @staticmethod
    def _get_user(context: Dict[str, Any]) -> User:
        """
        Get the user from the context using either a user_id or an aai_uid.
        Raises exception if no user is found.

        Args:
            context: context json  from the /recommendations endpoint request.

        Returns:
            User object model retrieved from the database.
        """

        user_id, aai_uid = context.get("user_id"), context.get("aai_uid")

        if not (user_id or aai_uid):
            raise UserCannotBeIdentified()

        return (
            User.objects(id=user_id).first()
            if user_id
            else User.objects(aai_uid=aai_uid).first()
        )

    @abstractmethod
    def _generate_recommendations(
        self, user: User, elastic_services: Tuple[int], search_data: SearchData
    ) -> List[int]:
        """
        Generate recommendation for logged-in user.

        Args:
            user: user for whom recommendation will be generated.
            elastic_services: item space from the Marketplace.
            search_data: search phrase and filters information for narrowing
             down an item space.

        Returns:
            recommended_services_ids: List of recommended services ids.
        """
