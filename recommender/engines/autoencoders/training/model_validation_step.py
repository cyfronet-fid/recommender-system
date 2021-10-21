# pylint: disable=line-too-long, disable=fixme
"""Autoencoder Model Validation Step."""

from typing import Tuple

from recommender.engines.base.base_steps import ModelValidationStep
from recommender.engines.autoencoders.training.model_evaluation_step import (
    METRICS,
)
from recommender.errors import PerformanceTooLowError

MODEL_IS_VALID = "model_is_valid"
MAX_LOSS_SCORE = "max_loss_score"


def check_performance(metrics, max_loss_score: int) -> None:
    """Check if the losses of the models are below certain threshold.
    Args:
        metrics: data from the AEModelEvaluationStep,
        max_loss_score: threshold used for validation.
    """

    for collection_name, metric in metrics.items():
        for split, loss in metric.items():
            if loss > max_loss_score:
                print(
                    f"Loss of the {collection_name} collection {split} was greater than {MAX_LOSS_SCORE}"
                )
                raise PerformanceTooLowError()


class AEModelValidationStep(ModelValidationStep):
    """Autoencoder model validation step"""

    def __init__(self, pipeline_config):
        super().__init__(pipeline_config)
        self.max_loss_score = self.resolve_constant(MAX_LOSS_SCORE, 2)

    def __call__(self, data=None) -> Tuple[object, object]:
        """Perform model validation consisting of checking:
        -> model performance regarding chosen metric.
        """
        metrics = data[METRICS]

        details = {MODEL_IS_VALID: False}
        check_performance(metrics, self.max_loss_score)
        details[MODEL_IS_VALID] = True

        return data, details
