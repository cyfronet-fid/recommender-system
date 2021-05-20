import random
from typing import Dict, Any, Tuple

from engine.panel_id_to_services_number_mapping import PANEL_ID_TO_K
from engine.pre_agent.pre_agent import InvalidRecommendationPanelIDError, _services_to_ids
from errors import TooSmallRecommendationSpace
from models import User, SearchData
from services.fts import retrieve_services


class BaseAgentRecommender:
    """
    Base Recommender class with basic functionality
    """

    def __init__(self) -> None:
        """
        Initialization function.
        """

        pass

    def call(self, context: Dict[str, Any]) -> Tuple[int]:
        """
        This function allows for getting recommended services for the
        recommendation endpoint based on recommendation context.

        Args:
            context: json dict from the /recommendations endpoint request.

        Returns:
            Tuple of recommended services ids.
        """

        self._load_models()
        K = self._get_K(context)
        user = self._get_user(context)
        search_data = context.get("search_data")

        if user is not None:
            return self._for_logged_user(user, search_data, K)
        else:
            return self._for_anonymous_user(search_data, K)

    def _load_models(self) -> None:
        """
        It loads model or models needed for recommendations and raise
         exception if it is not available in the database.
         """
        pass

    def _get_K(self, context: Dict[str, Any]) -> int:
        """
        Get the K constant from the context.

        Args:
            context: context json  from the /recommendations endpoint request.

        Returns:
            K constant.
        """

        K = PANEL_ID_TO_K.get(context["panel_id"])
        if K is None:
            raise InvalidRecommendationPanelIDError
        return K

    def _get_user(self, context: Dict[str, Any]) -> User:
        """
        Get the user from the context.

        Args:
            context: context json  from the /recommendations endpoint request.

        Returns:
            User.
        """

        user = None
        if context.get("user_id"):
            user = User.objects(id=context.get("user_id")).first()

        return user

    def _for_logged_user(self, user: User, search_data: SearchData, k: int) -> Tuple[int]:
        """
        Generate recommendation for logged user

        Args:
            user: user for whom recommendation will be generated.
            search_data: search phrase and filters information for narrowing down an item space.

        Returns:
            Tuple of recommended services ids.
        """
        pass

    def _for_anonymous_user(self, search_data: SearchData, K: int) -> Tuple[int]:
        """
        Generate recommendation for anonymous user

        Args:
            search_data: search phrase and filters information for narrowing down an item space.

        Returns:
            Tuple of recommended services ids.
        """

        candidate_services = list(retrieve_services(search_data))
        if len(candidate_services) < K:
            raise TooSmallRecommendationSpace
        recommended_services = random.sample(list(candidate_services), K)
        return _services_to_ids(recommended_services)
