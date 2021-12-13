# pylint: disable=missing-module-docstring, missing-class-docstring, invalid-name, fixme

from typing import Tuple

import numpy as np

from recommender.engines.base.base_steps import ModelValidationStep
from recommender.errors import InferenceTooSlowError, MeanRewardTooLowError
from logger_config import get_logger

TIME_UPPER_BOUND = "time_upper_bound"
REWARD_LOWER_BOUND = "reward_upper_bound"

logger = get_logger(__name__)


class RLModelValidationStep(ModelValidationStep):
    def __init__(self, config):
        super().__init__(config)
        self.time_upper_bound = self.resolve_constant(TIME_UPPER_BOUND, 1.0)
        self.reward_lower_bound = self.resolve_constant(REWARD_LOWER_BOUND, 0.0)

    def __call__(self, data=None) -> Tuple[None, dict]:
        metrics = data
        mean_recommendation_time = np.mean(metrics["recommendation_durations"])
        mean_reward = np.mean(metrics["rewards"])

        if mean_reward < self.reward_lower_bound:
            raise MeanRewardTooLowError()

        if mean_recommendation_time > self.time_upper_bound:
            raise InferenceTooSlowError()

        # TODO: Unify metrics checking across all pipelines
        return None, {"reward_check_passed": True, "recommendation_time_passed": True}
