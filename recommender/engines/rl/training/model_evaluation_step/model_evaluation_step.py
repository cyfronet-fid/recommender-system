# pylint: disable=fixme, missing-module-docstring, missing-class-docstring
# pylint: disable= invalid-name, too-many-locals, too-many-instance-attributes

import random
from time import time
from typing import Tuple

import pandas as pd

from recommender.engines.constants import DEVICE
from recommender.engines.rl.ml_components.reward_mapping import (
    TRANSITION_REWARDS_CSV_PATH,
)
from recommender.engines.autoencoders.ml_components.embedder import (
    Embedder,
    USER_EMBEDDER,
    SERVICE_EMBEDDER,
)
from recommender.engines.autoencoders.ml_components.normalizer import (
    Normalizer,
    NormalizationMode,
)
from recommender.engines.base.base_steps import ModelEvaluationStep
from recommender.engines.rl.ml_components.reward_encoder import RewardEncoder
from recommender.engines.rl.ml_components.state_encoder import StateEncoder
from recommender.engines.rl.ml_components.service_selector import ServiceSelector
from recommender.engines.rl.ml_components.synthetic_dataset.rewards import (
    synthesize_reward,
)
from recommender.engines.rl.utils import create_state
from recommender.models import Service, SearchData
from recommender.engines.rl.ml_components.synthetic_dataset.service_engagement import (
    approx_service_engagement,
)


TIME_MEASUREMENT_SAMPLES = "time_measurement_samples"


# TODO: Purge all database accesses and replace
# TODO: with data given by the data extraction step
class RLModelEvaluationStep(ModelEvaluationStep):
    def __init__(self, config):
        super().__init__(config)
        self.time_measurement_samples = self.resolve_constant(
            TIME_MEASUREMENT_SAMPLES, 50
        )
        self.device = self.resolve_constant(DEVICE, "cpu")

        self.user_embedder = Embedder.load(version=USER_EMBEDDER)
        self.service_embedder = Embedder.load(version=SERVICE_EMBEDDER)

        self.state_encoder = StateEncoder(
            user_embedder=self.user_embedder,
            service_embedder=self.service_embedder,
        )
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

    def _evaluate_reward(self, actor, sarses):
        # TODO: Later add more than 1 step per episode for evaluation
        rewards = []

        simulation_start = time()
        for sars in sarses:
            state = sars.state
            user = state.user
            service_ids = self._get_recommendation(actor, state)
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
        simulation_end = time()

        return rewards, simulation_end - simulation_start

    def _evaluate_recommendation_time(self, actor, sarses):
        recommendation_durations = []

        for _ in range(self.time_measurement_samples):
            start = time()
            example_user = random.choice(sarses).state.user
            example_state = create_state(example_user, SearchData())
            self._get_recommendation(actor, example_state)
            end = time()
            recommendation_durations.append(end - start)

        return recommendation_durations

    def _get_recommendation(self, actor, state):
        encoded_state = self.state_encoder([state])
        weights = actor(encoded_state).squeeze(0)
        mask = encoded_state[-1][0]
        chosen_services = self.service_selector(weights, mask)
        return chosen_services

    def __call__(self, data=None) -> Tuple[dict, dict]:
        actor, sarses = data
        actor.eval()

        rewards, simulation_duration = self._evaluate_reward(actor, sarses)
        recommendation_durations = self._evaluate_recommendation_time(actor, sarses)

        return {
            "recommendation_durations": recommendation_durations,
            "rewards": rewards,
        }, {
            "simulation_duration": simulation_duration,
        }


# if __name__ == '__main__':
# connect(host=DevelopmentConfig.MONGODB_HOST)
# actor = Actor(3, )
# config = {
#     ModelEvaluationStep.__name__: {
#         TIME_MEASUREMENT_SAMPLES: 50
#     },
# }
#
# x = RLModelEvaluationStep(config)
# print(x())
#
# disconnect()
