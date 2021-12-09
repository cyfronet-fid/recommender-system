# pylint: disable=missing-module-docstring, missing-class-docstring, invalid-name, not-callable

from time import time
from typing import Tuple, List

from recommender.engines.autoencoders.ml_components.embedder import (
    Embedder,
    SERVICE_EMBEDDER,
)
from recommender.engines.base.base_steps import DataExtractionStep
from recommender.engines.rl.ml_components.synthetic_dataset.rewards import (
    RewardGeneration,
)
from recommender.models import Sars, UserAction, Recommendation
from recommender.engines.rl.ml_components.sarses_generator import regenerate_sarses
from recommender.engines.rl.ml_components.synthetic_dataset.dataset import (
    generate_synthetic_sarses,
)

MIN_USER_ACTIONS = "min_user_actions"
MIN_RECOMMENDATIONS = "min_recommendations"
SYNTHETIC = "synthetic"
K = "K"
SYNTHETIC_PARAMS = "synthetic_params"
INTERACTIONS_RANGE = "interactions_range"
REWARD_GENERATION_MODE = "reward_generation_mode"
GENERATE_NEW = "generate_new"


class RLDataExtractionStep(DataExtractionStep):
    def __init__(self, pipeline_config):
        super().__init__(pipeline_config)
        self.min_user_actions = self.resolve_constant(MIN_USER_ACTIONS, 2500)
        self.min_recommendations = self.resolve_constant(MIN_RECOMMENDATIONS, 2500)
        self.K = self.resolve_constant(K)
        self.synthetic_params = self.resolve_constant(
            SYNTHETIC_PARAMS,
            {
                K: self.K,
                INTERACTIONS_RANGE: (1, 2),
                REWARD_GENERATION_MODE: RewardGeneration.COMPLEX,
            },
        )

    def __call__(self, data=None) -> Tuple[List[Sars], dict]:
        start = time()
        sarses = self._generate_proper_sarses()
        end = time()
        return sarses, {"sarses_generation_duration": end - start}

    def _generate_synthetic(self):
        no_of_user_actions = len(UserAction.objects)
        no_of_recommendations = len(Recommendation.objects)
        return (
            no_of_user_actions < self.min_user_actions
            and no_of_recommendations < self.min_recommendations
        )

    def _generate_proper_sarses(self):
        Sars.objects(__raw__={"action": {"$size": self.K}}).delete()

        if self._generate_synthetic():
            service_embedder = Embedder.load(version=SERVICE_EMBEDDER)
            return generate_synthetic_sarses(service_embedder, **self.synthetic_params)

        real_sarses = regenerate_sarses(multi_processing=True, verbose=False)
        real_sarses = real_sarses(__raw__={"action": {"$size": self.K}})

        return list(real_sarses)
