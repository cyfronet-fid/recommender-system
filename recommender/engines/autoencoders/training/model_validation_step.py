# pylint: disable=line-too-long
"""Autoencoder Model Validation Step."""

from typing import Tuple

from recommender.engines.base.base_steps import ModelValidationStep
from recommender.engines.constants import LOSS, METRICS
from recommender.errors import PerformanceTooLowError
from logger_config import get_logger

MODEL_IS_VALID = "model_is_valid"
MAX_LOSS_SCORE = "max_loss_score"

logger = get_logger(__name__)


def check_performance(metrics: dict, max_loss_score: int) -> None:
    """
    Check if the losses of the models are below certain threshold.
    Args:
        metrics: data from the AEModelEvaluationStep,
        max_loss_score: threshold used for validation.
    """
    for collection_name, metric in metrics.items():
        for split, data in metric.items():
            loss = data[LOSS]
            if loss > max_loss_score:
                raise PerformanceTooLowError(
                    f"Loss of the {collection_name} collection"
                    f"{split} was greater than {MAX_LOSS_SCORE}"
                )


class AEModelValidationStep(ModelValidationStep):
    """Autoencoder model validation step"""

    def __init__(self, pipeline_config):
        super().__init__(pipeline_config)
        self.max_loss_score = self.resolve_constant(MAX_LOSS_SCORE, 1.5)

    def __call__(self, data=None) -> Tuple[object, object]:
        """
        Perform model validation consisting of checking:
        -> model performance regarding chosen metric.
        """
        metrics = data[METRICS]
        details = {MODEL_IS_VALID: False}
        check_performance(metrics, self.max_loss_score)
        details[MODEL_IS_VALID] = True

        return data, details
