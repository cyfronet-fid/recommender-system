# pylint: disable=fixme, missing-module-docstring, missing-class-docstring
# pylint: disable= invalid-name, too-many-locals, too-many-instance-attributes, line-too-long

import random
from time import time
from typing import Tuple

import pandas as pd

from recommender.engines.constants import DEVICE
from recommender.engines.nlp_embedders.embedders import Services2tensorsEmbedder
from recommender.engines.rl.ml_components.reward_mapping import (
    TRANSITION_REWARDS_CSV_PATH,
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
from recommender.errors import InsufficientRecommendationSpaceError
from logger_config import get_logger


TIME_MEASUREMENT_SAMPLES = "time_measurement_samples"
logger = get_logger(__name__)


# TODO: Purge all database accesses and replace
# TODO: with data given by the data extraction step
class RLModelEvaluationStep(ModelEvaluationStep):
    def __init__(self, config):
        super().__init__(config)
        self.time_measurement_samples = self.resolve_constant(
            TIME_MEASUREMENT_SAMPLES, 50
        )
        self.device = self.resolve_constant(DEVICE, "cpu")

        self.state_encoder = StateEncoder()
        self.reward_encoder = RewardEncoder()
        self.service_selector = ServiceSelector()

        services = list(Service.objects.order_by("id"))
        self.services, self.index_id_map = Services2tensorsEmbedder()(services)

        self.transition_rewards_df = pd.read_csv(
            TRANSITION_REWARDS_CSV_PATH, index_col="source"
        )

    def _evaluate_reward(self, actor, sarses):
        # TODO: Later add more than 1 step per episode for evaluation
        rewards = []
        invalid_sars_ctr = 0

        simulation_start = time()
        for sars in sarses:
            state = sars.state
            user = state.user

            try:
                service_ids = self._get_recommendation(actor, state)
            except InsufficientRecommendationSpaceError as e:
                logger.debug(
                    "SARS with ID %s is invalid due to error '%s'", sars.id, e.message()
                )
                invalid_sars_ctr += 1
                continue

            services = Service.objects(id__in=service_ids)
            service_engagements = [
                approx_service_engagement(
                    user,
                    s,
                    state.services_history,
                    self.services,
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

        return rewards, simulation_end - simulation_start, invalid_sars_ctr

    def _evaluate_recommendation_time(self, actor, sarses):
        recommendation_durations = []

        for _ in range(self.time_measurement_samples):
            start = time()
            example_user = random.choice(sarses).state.user
            elastic_services = tuple(int(service.id) for service in Service.objects)  # TODO: so.. it works?
            example_state = create_state(example_user, elastic_services, SearchData())
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

        rewards, simulation_duration, invalid_sars_ctr = self._evaluate_reward(
            actor, sarses
        )
        recommendation_durations = self._evaluate_recommendation_time(actor, sarses)

        return {
            "recommendation_durations": recommendation_durations,
            "rewards": rewards,
        }, {
            "simulation_duration": f"{round(simulation_duration, 3)}s",
            "invalid_sarses": invalid_sars_ctr,
            "invalid_sarses/valid_sarses": f"{round(invalid_sars_ctr/len(sarses), 3) * 100}%",
        }
