# pylint: disable=invalid-name, no-self-use, no-member, too-few-public-methods, fixme

"""Implementation of the base agent recommmender. Every other agent should
 inherit from this one."""

from abc import ABC, abstractmethod
import random
from typing import Dict, Any, List

from recommender.engines.panel_id_to_services_number_mapping import PANEL_ID_TO_K
from recommender.errors import (
    InsufficientRecommendationSpace,
    InvalidRecommendationPanelIDError,
)
from recommender.models import User, SearchData
from recommender.services.fts import retrieve_services_for_recommendation


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
        search_data = self._get_search_data(context)

        if user is not None:
            return self._for_logged_user(user, search_data)
        return self._for_anonymous_user(search_data)

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

    def _get_search_data(self, context: Dict[str, Any]) -> SearchData:
        search_data = context.get("search_data")
        search_data.pop(
            "rating", None
        )  # We don't and we shouldn't take rating into consideration

        # To prevent q being None (for SearchPhraseEncoder it must be a string)
        search_data["q"] = search_data.get("q", "")

        search_data = SearchData(**search_data)

        return search_data

    @abstractmethod
    def _for_logged_user(self, user: User, search_data: SearchData) -> List[int]:
        """
        Generate recommendation for logged user

        Args:
            user: user for whom recommendation will be generated.
            search_data: search phrase and filters information for narrowing
             down an item space.

        Returns:
            recommended_services_ids: List of recommended services ids.
        """

    def _for_anonymous_user(self, search_data: SearchData) -> List[int]:
        """
        Generate recommendation for anonymous user

        Args:
            search_data: search phrase and filters information for narrowing
             down an item space.

        Returns:
            recommended_services_ids: List of recommended services ids.
        """

        candidate_services = list(retrieve_services_for_recommendation(search_data))
        if len(candidate_services) < self.K:
            raise InsufficientRecommendationSpace
        recommended_services = random.sample(list(candidate_services), self.K)
        recommended_services_ids = [s.id for s in recommended_services]

        return recommended_services_ids
