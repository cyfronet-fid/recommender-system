# pylint: disable=no-self-use, too-few-public-methods, no-member
# pylint: disable=invalid-name, consider-using-generator
"""RL Agent Recommender"""

from typing import Tuple, Optional
import torch

from recommender.models import User, SearchData
from recommender.errors import NoActorForK, MissingComponentError
from recommender.engine.utils import NoSavedModuleError, load_last_module
from recommender.engine.agents.base_agent import BaseAgent
from recommender.engine.agents.rl_agent.models.actor import (
    ACTOR_V1,
    ACTOR_V2,
    Actor,
)
from recommender.engine.agents.rl_agent.utils import create_state
from recommender.engine.agents.rl_agent.action_selector import ActionSelector
from recommender.engine.agents.rl_agent.preprocessing.state_encoder import StateEncoder


class RLAgent(BaseAgent):
    """
    Reinforcement learning Agent Recommender
    """

    def __init__(
        self,
        actor_v1: Optional[Actor] = None,
        actor_v2: Optional[Actor] = None,
        state_encoder: Optional[StateEncoder] = None,
        action_selector: Optional[ActionSelector] = None,
    ) -> None:
        super().__init__()
        self.actor_v1 = actor_v1
        self.actor_v2 = actor_v2
        self.state_encoder = state_encoder
        self.action_selector = action_selector

    def _load_models(self) -> None:
        """
        It loads model or models needed for recommendations and raise
         exception if it is not available in the database
        """

        try:
            self.actor_v1 = self.actor_v1 or load_last_module(ACTOR_V1)
            self.actor_v2 = self.actor_v2 or load_last_module(ACTOR_V2)
        except NoSavedModuleError as no_saved_module:
            raise MissingComponentError(
                "Missing Actor Model in the database"
            ) from no_saved_module

        self.actor_v1.eval()
        self.actor_v2.eval()

        # TODO: move this back to init (because now it takes time during
        #  each request!!!)
        self.state_encoder = self.state_encoder or StateEncoder()
        self.action_selector = self.action_selector or ActionSelector()

    def _for_logged_user(
        self, user: User, search_data: SearchData, K: int
    ) -> Tuple[int]:
        """
        Generate recommendation for logged user

        Args:
            user: user for whom recommendation will be generated.
            search_data: search phrase and filters information for narrowing
             down the item space.

        Returns:
            Tuple of recommended service ids.
        """

        print("FOR LOGGED")

        actor = self._choose_actor(K)

        state = create_state(user, search_data)
        state_tensors = self.state_encoder(state)
        weights_tensor = self._use_actor(state_tensors, actor)
        print(f"weights_tensor: {weights_tensor}")
        recommended_service_ids = self.action_selector(
            K, weights_tensor, user, search_data
        )

        return recommended_service_ids

    def _choose_actor(self, K: int) -> torch.nn.Module:
        """Choose appropriate model depending on the demanded recommended
        services number"""

        if K == 3:
            actor = self.actor_v1
        elif K == 2:
            actor = self.actor_v2
        else:
            raise NoActorForK

        return actor

    def _use_actor(self, state_tensors, actor):
        """
        It performs following steps:
            -> prepares input for actor,
            -> handle inference
            -> prepare output.
        """

        state_tensors = tuple([t.unsqueeze(0) for t in state_tensors])

        with torch.no_grad():
            weights_tensor = actor(state_tensors)

        weights_tensor = weights_tensor.squeeze(0)

        return weights_tensor
