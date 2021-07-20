# pylint: disable=no-self-use, too-few-public-methods, no-member
# pylint: disable=invalid-name, consider-using-generator, fixme
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
from recommender.engine.agents.rl_agent.service_selector import ServiceSelector
from recommender.engine.agents.rl_agent.preprocessing.state_encoder import StateEncoder

RL_AGENT = "rl_agent"


class RLAgent(BaseAgent):
    """
    Reinforcement learning Agent Recommender
    """

    def __init__(
        self,
        actor_v1: Optional[Actor] = None,
        actor_v2: Optional[Actor] = None,
        state_encoder: Optional[StateEncoder] = None,
        service_selector: Optional[ServiceSelector] = None,
    ) -> None:
        self.actor_v1 = actor_v1
        self.actor_v2 = actor_v2
        self.state_encoder = state_encoder
        self.service_selector = service_selector
        super().__init__()

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

        self.state_encoder = self.state_encoder or StateEncoder()
        self.service_selector = self.service_selector or ServiceSelector()

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

        actor = self._choose_actor(K)

        state = create_state(user, search_data)
        state_tensors = self.state_encoder([state])
        weights_tensor = self._use_actor(state_tensors, actor)

        search_data_mask = state_tensors[2][0]

        recommended_service_ids = self.service_selector(
            K, weights_tensor, search_data_mask
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

        with torch.no_grad():
            weights_tensor = actor(state_tensors)

        weights_tensor = weights_tensor.squeeze(0)

        return weights_tensor
