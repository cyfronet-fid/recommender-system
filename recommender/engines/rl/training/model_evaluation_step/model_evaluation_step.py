# pylint: disable=fixme, missing-module-docstring, missing-class-docstring
# pylint: disable= invalid-name, too-many-locals, too-many-instance-attributes

import random
from time import time
from typing import Tuple

import pandas as pd

from recommender.engine.agents.rl_agent.reward_mapping import (
    TRANSITION_REWARDS_CSV_PATH,
)
from recommender.engines.autoencoders.ml_components.embedder import Embedder
from recommender.engines.autoencoders.ml_components.normalizer import (
    Normalizer,
    NormalizationMode,
)
from recommender.engines.base.base_steps import ModelEvaluationStep
from recommender.engines.rl.ml_components.reward_encoder import RewardEncoder
from recommender.engines.rl.ml_components.state_encoder import StateEncoder
from recommender.engines.rl.ml_components.service_selector import ServiceSelector
from recommender.engines.rl.utils import create_state
from recommender.models import User, Service, SearchData
from recommender.services.synthetic_dataset.rewards import synthesize_reward
from recommender.services.synthetic_dataset.service_engagement import (
    approx_service_engagement,
)


# TODO: Purge all database accesses and replace
# TODO: with data given by the data preparation step
class RLModelEvaluationStep(ModelEvaluationStep):
    def __init__(self, config):
        super().__init__(config)
        self.time_measurement_samples = self.resolve_constant(
            "time_measurement_samples", 50
        )

        self.users = User.objects()

        self.user_embedder = Embedder.load(version="user")
        self.service_embedder = Embedder.load(version="service")

        self.state_encoder = StateEncoder(self.user_embedder, self.service_embedder)
        self.reward_encoder = RewardEncoder()

        self.service_selector = ServiceSelector(self.service_embedder)

        self.normalized_services, self.index_id_map = self._embed_and_normalize()

        self.transition_rewards_df = pd.read_csv(
            TRANSITION_REWARDS_CSV_PATH, index_col="source"
        )

    def _embed_and_normalize(self):
        service_embedded_tensors, index_id_map = self.service_embedder(
            Service.objects.order_by("id"), use_cache=False, save_cache=False
        )
        normalizer = Normalizer(mode=NormalizationMode.NORM_WISE)
        normalized_services, _ = normalizer(service_embedded_tensors)

        return normalized_services, index_id_map

    def _evaluate_reward(self, actor):
        # TODO: Later add more than 1 step per episode for evaluation
        actor.eval()
        rewards = []
        simulation_step_durations = []

        simulation_start = time()
        for user in self.users:
            step_start = time()
            state = create_state(user, SearchData())
            encoded_state = self.state_encoder([state])
            weights = actor(encoded_state).squeeze(0)
            mask = encoded_state[-1][0]
            service_ids = self.service_selector(weights, mask)
            services = Service.objects(id__in=service_ids)
            service_engagements = [
                approx_service_engagement(
                    user,
                    s,
                    state.services_history,
                    self.normalized_services,
                    self.index_id_map,
                )
                for s in services
            ]

            raw_reward = [
                synthesize_reward(self.transition_rewards_df, engagement)
                for engagement in service_engagements
            ]
            encoded_reward = self.reward_encoder([raw_reward])
            rewards.append(encoded_reward[0].item())
            step_end = time()
            simulation_step_durations.append(step_end - step_start)
        simulation_end = time()

        return rewards, simulation_end - simulation_start, simulation_step_durations

    def _evaluate_recommendation_time(self, actor):
        recommendation_durations = []

        for _ in range(self.time_measurement_samples):
            start = time()
            example_user = random.choice(self.users)
            example_state = create_state(example_user, SearchData())
            encoded_state = self.state_encoder([example_state])
            weights = actor(encoded_state).squeeze(0)
            mask = encoded_state[-1][0]
            self.service_selector(weights, mask)
            end = time()
            recommendation_durations.append(end - start)

        return recommendation_durations

    def __call__(self, data=None) -> Tuple[dict, dict]:
        actor = data
        actor.eval()

        rewards, simulation_duration, simulation_step_durations = self._evaluate_reward(
            actor
        )
        recommendation_durations = self._evaluate_recommendation_time(actor)

        return {
            "recommendation_durations": recommendation_durations,
            "rewards": rewards,
        }, {
            "simulation_step_durations": simulation_step_durations,
            "simulation_duration": simulation_duration,
        }
