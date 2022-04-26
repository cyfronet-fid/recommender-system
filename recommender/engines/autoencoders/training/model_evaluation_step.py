# pylint: disable=line-too-long, too-many-locals
"""Autoencoder Model Evaluation Step."""

from typing import Tuple
import time
import torch
from torch.utils.data import DataLoader
from recommender.engines.base.base_steps import ModelEvaluationStep
from recommender.engines.constants import (
    DEVICE,
    METRICS,
    LOSS,
    ACCURACY,
    VERBOSE,
)
from recommender.engines.autoencoders.training.model_training_step import (
    MODEL,
    DATASET,
    evaluate_autoencoder,
    autoencoder_loss_function,
)
from recommender.engines.autoencoders.training.data_preparation_step import (
    TRAIN,
    VALID,
    TEST,
)
from recommender.engines.autoencoders.training.data_extraction_step import (
    USERS,
    SERVICES,
)
from recommender.engines.metadata_creators import accuracy_function
from logger_config import get_logger

logger = get_logger(__name__)

BATCH_SIZE = "batch_size"


class AEModelEvaluationStep(ModelEvaluationStep):
    """Autoencoder model evaluation step"""

    def __init__(self, pipeline_config):
        super().__init__(pipeline_config)
        self.device = self.resolve_constant(DEVICE, torch.device("cpu"))
        self.batch_size = self.resolve_constant(BATCH_SIZE, 128)
        self.verbose = self.resolve_constant(VERBOSE, True)

    def __call__(self, data=None) -> Tuple[object, dict]:

        metrics = {
            USERS: {TRAIN: {}, VALID: {}, TEST: {}},
            SERVICES: {TRAIN: {}, VALID: {}, TEST: {}},
        }

        start_evaluation = time.time()

        for collection_name, datasets in data.items():
            model = datasets[MODEL]
            for split, dataset in datasets[DATASET].items():
                dataloader = DataLoader(
                    dataset, batch_size=self.batch_size, shuffle=True
                )
                loss, acc = evaluate_autoencoder(
                    model,
                    dataloader,
                    autoencoder_loss_function,
                    accuracy_function,
                    self.device,
                )
                metrics[collection_name][split][LOSS] = loss
                metrics[collection_name][split][ACCURACY] = acc

        if self.verbose:
            logger.info(
                "Evaluation total duration: %ss",
                round(time.time() - start_evaluation, 3),
            )

        details = {METRICS: metrics}
        data[METRICS] = metrics

        return data, details
