# pylint: disable-all

import pytest

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn import CosineEmbeddingLoss
from torch.optim import Adam

from definitions import LOG_DIR
from tests.factories.populate_database import populate_users_and_services
from recommender.engines.constants import DEVICE, WRITER, VERBOSE

from recommender.engines.autoencoders.training.data_extraction_step import (
    DataExtractionStep,
)
from recommender.engines.autoencoders.training.data_validation_step import (
    DataValidationStep,
    LEAST_NUM_OF_USR_SRV,
)
from recommender.engines.autoencoders.training.data_preparation_step import (
    DataPreparationStep,
    TRAIN_DS_SIZE,
    VALID_DS_SIZE,
)
from recommender.engines.autoencoders.training.model_training_step import (
    ModelTrainingStep,
    USER_BATCH_SIZE,
    SERVICE_BATCH_SIZE,
    USER_EMBEDDING_DIM,
    SERVICE_EMBEDDING_DIM,
    EPOCHS,
    OPTIMIZER,
    LR,
    LOSS_FUNCTION,
)
from recommender.engines.autoencoders.training.model_evaluation_step import (
    ModelEvaluationStep,
    BATCH_SIZE,
)
from recommender.engines.autoencoders.training.model_validation_step import (
    ModelValidationStep,
    MAX_LOSS_SCORE,
)


@pytest.fixture
def generate_data():
    populate_users_and_services(
        common_services_number=9,
        no_one_services_number=9,
        users_number=10,
        k_common_services_min=5,
        k_common_services_max=7,
    )


@pytest.fixture
def pipeline_config():
    config = {
        DEVICE: torch.device("cpu"),
        WRITER: SummaryWriter(log_dir=LOG_DIR),
        VERBOSE: False,
        LOSS_FUNCTION: CosineEmbeddingLoss(),
        DataExtractionStep.__name__: {},
        DataValidationStep.__name__: {LEAST_NUM_OF_USR_SRV: 1},
        DataPreparationStep.__name__: {TRAIN_DS_SIZE: 0.6, VALID_DS_SIZE: 0.2},
        ModelTrainingStep.__name__: {
            USER_BATCH_SIZE: 128,
            SERVICE_BATCH_SIZE: 128,
            USER_EMBEDDING_DIM: 32,
            SERVICE_EMBEDDING_DIM: 64,
            EPOCHS: 500,
            OPTIMIZER: Adam,
            LR: 0.01,
        },
        ModelEvaluationStep.__name__: {BATCH_SIZE: 128},
        ModelValidationStep.__name__: {MAX_LOSS_SCORE: 2},
    }

    return config
