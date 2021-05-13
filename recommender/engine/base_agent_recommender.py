# pylint: disable=invalid-name, no-self-use, no-member, too-few-public-methods

"""Implementation of the base agent recommmender. Every other agent should
 inherit from this one."""

from abc import ABC, abstractmethod
import random
from typing import Dict, Any, List

from recommender.errors import (
    InvalidRecommendationPanelIDError,
    InsufficientRecommendationSpace,
)
from recommender.engine.panel_id_to_services_number_mapping import PANEL_ID_TO_K
from recommender.models import User, SearchData, Service
from recommender.services.fts import retrieve_services


class BaseAgentRecommender(ABC):
    """
    Base Recommender class with basic functionality
    """

    def __init__(self) -> None:
        """
        Initialization function.
        """

    def call(self, context: Dict[str, Any]) -> List[int]:
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
        return self._for_anonymous_user(search_data, K)

    @abstractmethod
    def _load_models(self) -> None:
        """
        It loads model or models needed for recommendations and raise
         exception if it is not available in the database.
        """

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

    @abstractmethod
    def _for_logged_user(
        self, user: User, search_data: SearchData, k: int
    ) -> List[int]:
        """
        Generate recommendation for logged user

        Args:
            user: user for whom recommendation will be generated.
            search_data: search phrase and filters information for narrowing
             down an item space.

        Returns:
            List of recommended services ids.
        """

    def _for_anonymous_user(self, search_data: SearchData, K: int) -> List[int]:
        """
        Generate recommendation for anonymous user

        Args:
            search_data: search phrase and filters information for narrowing
             down an item space.

        Returns:
            Tuple of recommended services ids.
        """

        candidate_services = list(retrieve_services(search_data))
        candidate_services = self._fill_candidate_services(candidate_services, K)
        recommended_services = random.sample(list(candidate_services), K)
        return [s.id for s in recommended_services]

    def _fill_candidate_services(
        self, candidate_services, required_services_no, accessed_services_ids=None
    ):
        """Fallback in case of len of the candidate_services being too small"""

        if len(candidate_services) < required_services_no:
            missing_n = required_services_no - len(candidate_services)

            # We don't want to recommend same service multiple times
            # in one recommendation panel
            forbidden_ids = {s.id for s in candidate_services}

            # We don't want to recommend a service if it already
            # has been ordered by a user
            if accessed_services_ids is not None:
                forbidden_ids = forbidden_ids | set(accessed_services_ids)

            allowed_services = list(Service.objects(id__nin=forbidden_ids))

            if len(allowed_services) >= missing_n:
                sampled = random.sample(allowed_services, missing_n)
                candidate_services += sampled
            else:
                raise InsufficientRecommendationSpace

        return candidate_services
