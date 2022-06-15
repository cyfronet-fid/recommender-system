# pylint: disable=missing-module-docstring, missing-class-docstring, too-few-public-methods
# pylint: disable=too-many-instance-attributes, no-self-use, line-too-long

from typing import Tuple

import torch

from recommender.engines.panel_id_to_services_number_mapping import K_TO_PANEL_ID
from recommender.engines.rl.utils import create_state
from recommender.engines.base.base_inference_component import MLEngineInferenceComponent
from recommender.engines.rl.ml_components.actor import Actor
from recommender.engines.rl.ml_components.state_encoder import StateEncoder
from recommender.engines.rl.ml_components.service_selector import ServiceSelector
from recommender.models import User, SearchData


class RLInferenceComponent(MLEngineInferenceComponent):
    """
    Recommender engine that provides logged-in users with personalized recommendations in a given context.
    """

    engine_name = "RL"

    def __init__(self, K: int, exploration: bool = False, act_noise: float = 0.0):
        self.exploration = exploration
        self.act_noise = act_noise
        super().__init__(K)

    def _load_models(self) -> None:
        """It loads model or models needed for recommendations and raise
        exception if it is not available in the database.
        """

        version = K_TO_PANEL_ID.get(self.K)
        self.actor = Actor.load(version=version)
        self.actor.eval()
        self.state_encoder = StateEncoder()
        self.service_selector = ServiceSelector()

    def _generate_recommendations(
        self, user: User, elastic_services: Tuple[int], search_data: SearchData
    ) -> Tuple[int]:
        """Generate recommendation for logged user.

        Args:
            user: user for whom recommendation will be generated.
            elastic_services: item space from the Marketplace.
            search_data: search phrase and filters information for narrowing
             down an item space.

        Returns:
            service_ids: List of recommended services ids.
        """

        state = create_state(user, elastic_services, search_data)
        state_tensors = self.state_encoder([state])
        weights_tensor = self._get_weights(state_tensors)
        services_mask = self._get_service_mask(state_tensors)
        service_ids = self.service_selector(weights_tensor, services_mask)
        return service_ids

    @staticmethod
    def _get_service_mask(state_tensors):
        return state_tensors[2][0]

    def _get_weights(self, state_tensors):
        with torch.no_grad():
            weights_tensor = self.actor(state_tensors).squeeze(0)

        if self.exploration:
            weights_tensor = self._add_noise(weights_tensor)

        return weights_tensor

    def _add_noise(self, action):
        noise = torch.randn_like(action)
        noise *= self.act_noise
        noised_action = action + noise
        return noised_action.clamp(max=self.actor.act_max, min=self.actor.act_min)
