# pylint: disable-all

import pytest
import torch
from torch.nn import CosineEmbeddingLoss
from torch.optim import Adam
from recommender.engines.autoencoders.inference.embedding_component import (
    EmbeddingComponent,
)
from recommender.engines.autoencoders.ml_components.autoencoder import AutoEncoder
from recommender.engines.autoencoders.ml_components.embedder import (
    Embedder,
    USER_EMBEDDER,
    SERVICE_EMBEDDER,
)
from recommender.engines.autoencoders.training.data_preparation_step import (
    precalc_users_and_service_tensors,
    TRAIN_DS_SIZE,
    VALID_DS_SIZE,
)
from recommender.engines.autoencoders.training.data_validation_step import (
    LEAST_NUM_OF_USR_SRV,
)
from recommender.engines.autoencoders.training.model_evaluation_step import BATCH_SIZE
from recommender.engines.autoencoders.training.model_training_step import (
    ENCODER_LAYER_SIZES,
    DECODER_LAYER_SIZES,
    USER_EMBEDDING_DIM,
    SERVICE_EMBEDDING_DIM,
    LOSS_FUNCTION,
    USER_BATCH_SIZE,
    SERVICE_BATCH_SIZE,
    EPOCHS,
    OPTIMIZER,
    LR,
)
from recommender.engines.autoencoders.training.model_validation_step import (
    MAX_LOSS_SCORE,
)
from recommender.engines.base.base_steps import (
    ModelTrainingStep,
    DataExtractionStep,
    DataValidationStep,
    DataPreparationStep,
    ModelEvaluationStep,
    ModelValidationStep,
)
from recommender.engines.constants import DEVICE, WRITER, VERBOSE
from recommender.models import User, Service
from tests.factories.populate_database import populate_users_and_services


@pytest.fixture
def generate_data(mongo):
    populate_users_and_services(
        common_services_no=9,
        unordered_services_no=9,
        total_users=5,
        k_common_services_min=5,
        k_common_services_max=7,
    )


@pytest.fixture
def embedding_dims():
    user_embedding_dim = 64
    service_embedding_dim = 128
    return user_embedding_dim, service_embedding_dim


@pytest.fixture
def ae_pipeline_config(embedding_dims):
    user_embedding_dim, service_embedding_dim = embedding_dims
    config = {
        DEVICE: torch.device("cpu"),
        WRITER: None,
        VERBOSE: False,
        LOSS_FUNCTION: CosineEmbeddingLoss(),
        DataExtractionStep.__name__: {},
        DataValidationStep.__name__: {LEAST_NUM_OF_USR_SRV: 1},
        DataPreparationStep.__name__: {TRAIN_DS_SIZE: 0.6, VALID_DS_SIZE: 0.2},
        ModelTrainingStep.__name__: {
            ENCODER_LAYER_SIZES: (128, 64),
            DECODER_LAYER_SIZES: (64, 128),
            USER_BATCH_SIZE: 128,
            SERVICE_BATCH_SIZE: 128,
            USER_EMBEDDING_DIM: user_embedding_dim,
            SERVICE_EMBEDDING_DIM: service_embedding_dim,
            EPOCHS: 500,
            OPTIMIZER: Adam,
            LR: 0.01,
        },
        ModelEvaluationStep.__name__: {BATCH_SIZE: 128},
        ModelValidationStep.__name__: {MAX_LOSS_SCORE: 2},
    }

    return config


@pytest.fixture
def mock_autoencoders_pipeline_exec(mongo, ae_pipeline_config):
    precalc_users_and_service_tensors()

    USER_ONE_HOT_DIM = len(User.objects.first().one_hot_tensor)

    user_autoencoder_mock = AutoEncoder(
        USER_ONE_HOT_DIM,
        ae_pipeline_config[ModelTrainingStep.__name__][USER_EMBEDDING_DIM],
    )
    user_embedder = Embedder(user_autoencoder_mock)

    SERVICE_ONE_HOT_DIM = len(Service.objects.first().one_hot_tensor)

    service_autoencoder_mock = AutoEncoder(
        SERVICE_ONE_HOT_DIM,
        ae_pipeline_config[ModelTrainingStep.__name__][SERVICE_EMBEDDING_DIM],
    )
    service_embedder = Embedder(service_autoencoder_mock)

    user_embedder.save(USER_EMBEDDER)
    service_embedder.save(SERVICE_EMBEDDER)


@pytest.fixture
def embedding_exec(mock_autoencoders_pipeline_exec):
    embedding_component = EmbeddingComponent()
    embedding_component()
