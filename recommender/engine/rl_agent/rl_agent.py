from typing import Tuple

import torch

from engine.base_agent_recommender import BaseAgentRecommender
from engine.pre_agent.pre_agent import _services_to_ids
from engine.rl_agent.preprocessing.state_embedder import StateEmbedder
from models import User, SearchData, State


class RLAgentRecommender(BaseAgentRecommender):
    """
    Reinforcement learning Agent Recommender
    """

    def __init__(self, actor_model_2=None, actor_model_3=None, state_embedder=None, action_selector=None) -> None:
        super().__init__()
        self.actor_model_2, self.actor_model_3 = actor_model_2, actor_model_3
        self.state_embedder = StateEmbedder()
        self.action_selector = ActionSelector()
        # TODO: loading from database if None

    def _load_models(self) -> None:
        """
        It loads model or models needed for recommendations and raise
         exception if it is not available in the database
        """

        # TODO: implement lazy models (and maybe other things) loading
        self.actor_model_2, self.actor_model_3 = None, None
        pass

    def _for_logged_user(self, user: User, search_data: SearchData, k: int) -> Tuple[int]:
        """
        Generate recommendation for logged user

        Args:
            user: user for whom recommendation will be generated.
            search_data: search phrase and filters information for narrowing down an item space.

        Returns:
            Tuple of recommended services ids.
        """

        actor_model = self._choose_actor_model(k)

        state = self._create_state(user, search_data)
        state_tensors = self.state_embedder(state)

        weights_tensor = actor_model(*state_tensors)
        recommended_services = self.action_selector(weights_tensor, user, search_data) # TODO: jednak ids
        recommended_services_ids = _services_to_ids(recommended_services)

        return recommended_services_ids

    def _choose_actor_model(self, k: int) -> torch.nn.Module:
        """Choose appropriate model depending on the demanded recommended
         services number"""

        # TODO: implement
        pass

    def _create_state(self, user: User, search_data: SearchData) -> State:
        """
        Get needed information from context and create state.
        Args:
            context: context json  from the /recommendations endpoint request.

        Returns:
            state: State containing information about user and search_data
        """

        # TODO: implement
        pass
