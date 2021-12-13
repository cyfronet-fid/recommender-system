# pylint: disable=missing-module-docstring, missing-class-docstring, invalid-name

import itertools
from typing import Tuple

from recommender.engines.base.base_steps import DataValidationStep
from recommender.errors import DataSetTooSmallError, ImbalancedDatasetError
from recommender.models import Sars
from logger_config import get_logger

MIN_EMPTY_TO_NON_EMPTY_RATIO = "min_empty_to_non_empty_ratio"
MIN_SARSES = "min_sarses"

logger = get_logger(__name__)


class RLDataValidationStep(DataValidationStep):
    def __init__(self, pipeline_config):
        super().__init__(pipeline_config)
        self.minimum_sarses = self.resolve_constant(MIN_SARSES, 500)
        self.min_empty_to_non_empty_ratio = self.resolve_constant(
            MIN_EMPTY_TO_NON_EMPTY_RATIO, float("inf")
        )

    def _check_for_minimum_sarses(self, sarses):
        no_of_sarses = len(sarses)

        if len(sarses) < self.minimum_sarses:
            raise DataSetTooSmallError()

        return no_of_sarses

    def _check_for_dataset_balance(self, sarses):
        empty = len(list(filter(self._empty_reward, sarses)))
        non_empty = len(sarses) - empty

        ratio = empty / (non_empty + 1)

        if ratio > self.min_empty_to_non_empty_ratio:
            raise ImbalancedDatasetError()

        return ratio

    @staticmethod
    def _empty_reward(sars):
        return len(list(itertools.chain(*sars.reward))) == 0

    def __call__(self, data=None) -> Tuple[Sars, dict]:
        sarses = data
        no_of_sarses = self._check_for_minimum_sarses(sarses)
        empty_to_non_empty_ratio = self._check_for_dataset_balance(sarses)

        return sarses, {
            "no_of_sarses": no_of_sarses,
            "empty_to_non_empty_ratio": empty_to_non_empty_ratio,
        }
