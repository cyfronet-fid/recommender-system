# pylint: disable=line-too-long,
"""Autoencoder Model Evaluation Step."""

from typing import Tuple
import torch
from torch.utils.data import DataLoader

from recommender.engines.base.base_steps import ModelEvaluationStep
from recommender.engines.constants import DEVICE
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

BATCH_SIZE = "batch_size"
METRICS = "metrics"
LOSS = "loss"


class AEModelEvaluationStep(ModelEvaluationStep):
    """Autoencoder model evaluation step"""

    def __init__(self, pipeline_config):
        super().__init__(pipeline_config)
        self.device = self.resolve_constant(DEVICE, torch.device("cpu"))
        self.batch_size = self.resolve_constant(BATCH_SIZE, 128)

    def __call__(self, data=None) -> Tuple[object, dict]:

        metrics = {
            USERS: {TRAIN: {}, VALID: {}, TEST: {}},
            SERVICES: {TRAIN: {}, VALID: {}, TEST: {}},
        }

        for collection_name, datasets in data.items():
            model = datasets[MODEL]
            for split, dataset in datasets[DATASET].items():
                dataloader = DataLoader(
                    dataset, batch_size=self.batch_size, shuffle=True
                )
                loss = evaluate_autoencoder(
                    model, dataloader, autoencoder_loss_function, self.device
                )
                metrics[collection_name][split] = loss

        details = {METRICS: metrics}
        data[METRICS] = metrics

        return data, details
