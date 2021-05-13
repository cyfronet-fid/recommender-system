# pylint: disable=no-self-use, too-few-public-methods

"""RL Agent Recommender"""

import copy
from typing import Tuple, Dict, Any
import torch

from recommender.engine.pre_agent.models.common import NoSavedModuleError
from recommender.errors import UntrainedRLAgentError, NoActorModelForK
from recommender.engine.rl_agent.action_selector import ActionSelector
from recommender.engine.base_agent_recommender import BaseAgentRecommender
from recommender.engine.pre_agent.models import load_last_module
from recommender.engine.rl_agent.preprocessing.state_embedder import StateEmbedder
from recommender.models import User, SearchData, State
from recommender.services.services_history_generator import generate_services_history


class RLAgentRecommender(BaseAgentRecommender):
    """
    Reinforcement learning Agent Recommender
    """

    def __init__(
        self,
        actor_model_2=None,
        actor_model_3=None,
        state_embedder=None,
        action_selector=None,
    ) -> None:
        super().__init__()
        self.actor_model_2 = actor_model_2
        self.actor_model_3 = actor_model_3
        self.state_embedder = state_embedder or StateEmbedder()
        self.action_selector = action_selector or ActionSelector()

    def _load_models(self) -> None:
        """
        It loads model or models needed for recommendations and raise
         exception if it is not available in the database
        """

        if self.actor_model_2 is None:
            try:
                self.actor_model_2 = load_last_module(
                    ACTOR_MODEL_2
                )  # TODO: import this constant from actor_model.py
            except NoSavedModuleError as no_saved_module:
                raise UntrainedRLAgentError from no_saved_module

        if self.actor_model_3 is None:
            try:
                self.actor_model_3 = load_last_module(
                    ACTOR_MODEL_3
                )  # TODO: import this constant from actor_model.py
            except NoSavedModuleError as no_saved_module:
                raise UntrainedRLAgentError from no_saved_module

    def _for_logged_user(
        self, user: User, search_data: Dict[str, Any], k: int
    ) -> Tuple[int]:
        """
        Generate recommendation for logged user

        Args:
            user: user for whom recommendation will be generated.
            search_data: search phrase and filters information for narrowing
             down an item space.

        Returns:
            Tuple of recommended services ids.
        """

        actor_model = self._choose_actor_model(k)

        state = self._create_state(user, search_data)
        state_tensors = self.state_embedder(state)

        weights_tensor = actor_model(*state_tensors)
        recommended_services_ids = self.action_selector(
            weights_tensor, user, search_data
        )

        return recommended_services_ids

    def _choose_actor_model(self, k: int) -> torch.nn.Module:
        """Choose appropriate model depending on the demanded recommended
        services number"""

        if k == 2:
            actor_model = self.actor_model_2
        elif k == 3:
            actor_model = self.actor_model_3
        else:
            raise NoActorModelForK

        return actor_model

    def _create_state(self, user: User, search_data: Dict[str, Any]) -> State:
        """
        Get needed information from context and create state.
        Args:
            context: context json  from the /recommendations endpoint request.

        Returns:
            state: State containing information about user and search_data
        """

        search_data = copy.deepcopy(search_data)
        search_data.pop("rating")

        state = State(
            user=user,
            services_history=generate_services_history(user),
            last_search_data=SearchData(**search_data),
        )
        state.reload()

        return state
