# pylint: disable-all

import pytest
import torch
from torch.nn import BCELoss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from definitions import LOG_DIR
from recommender.engine.models.autoencoders import AutoEncoder
from recommender.engine.preprocessing import precalc_users_and_service_tensors
from recommender.engine.preprocessing.embedder import Embedder
from recommender.engines.base.base_steps import (
    DataExtractionStep,
    DataValidationStep,
    DataPreparationStep,
    ModelTrainingStep,
    ModelEvaluationStep,
    ModelValidationStep,
)
from recommender.engines.constants import DEVICE, WRITER, VERBOSE
from recommender.engines.ncf.ml_components.neural_collaborative_filtering import (
    NeuralCollaborativeFilteringModel,
    NEURAL_CF,
)
from recommender.engines.ncf.training.data_extraction_step import (
    MAX_USERS,
)
from recommender.engines.ncf.training.data_preparation_step import (
    TRAIN_DS_SIZE,
    VALID_DS_SIZE,
)
from recommender.engines.ncf.training.data_validation_step import (
    LEAST_N_ORDERS_PER_USER,
)
from recommender.engines.ncf.training.model_training_step import (
    BATCH_SIZE,
    SE,
    UE,
    LOSS_FUNCTION,
    MF_EMBEDDING_DIM,
    USER_IDS_EMBEDDING_DIM,
    SERVICE_IDS_EMBEDDING_DIM,
    MLP_LAYERS_SPEC,
    CONTENT_MLP_LAYERS_SPEC,
    EPOCHS,
    OPTIMIZER,
    OPTIMIZER_PARAMS,
)
from recommender.engines.ncf.training.model_validation_step import (
    MAX_EXECUTION_TIME,
    MIN_WEIGHTED_AVG_F1_SCORE,
    MAX_ITEMSPACE_SIZE,
)
from recommender.models import User, Service
from tests.factories.populate_database import populate_users_and_services


@pytest.fixture
def generate_data():
    populate_users_and_services(
        common_services_number=9,
        no_one_services_number=9,
        users_number=5,
        k_common_services_min=5,
        k_common_services_max=7,
    )


@pytest.fixture
def pipeline_config():
    config = {
        DEVICE: torch.device("cpu"),
        WRITER: SummaryWriter(log_dir=LOG_DIR),
        VERBOSE: False,
        BATCH_SIZE: 64,
        SE: 128,
        UE: 64,
        LOSS_FUNCTION: BCELoss(),
        DataExtractionStep.__name__: {MAX_USERS: None},
        DataValidationStep.__name__: {LEAST_N_ORDERS_PER_USER: 5},
        DataPreparationStep.__name__: {TRAIN_DS_SIZE: 0.6, VALID_DS_SIZE: 0.2},
        ModelTrainingStep.__name__: {
            MF_EMBEDDING_DIM: 64,
            USER_IDS_EMBEDDING_DIM: 64,
            SERVICE_IDS_EMBEDDING_DIM: 64,
            MLP_LAYERS_SPEC: (64, 32, 16, 8),
            CONTENT_MLP_LAYERS_SPEC: (128, 64, 32),
            EPOCHS: 500,
            OPTIMIZER: Adam,
            OPTIMIZER_PARAMS: {"lr": 0.01},
        },
        ModelEvaluationStep.__name__: {},
        ModelValidationStep.__name__: {
            # Below MAX_EXECUTION_TIME is so high because of slow CPU on the
            # remote github actions CI pipeline. It should be ofc abou 0.1 [s]
            MAX_EXECUTION_TIME: 100,
            MAX_ITEMSPACE_SIZE: 1000,
            MIN_WEIGHTED_AVG_F1_SCORE: 0.1,
        },
    }

    return config


@pytest.fixture
def mock_autoencoders_pipeline_exec(pipeline_config):
    precalc_users_and_service_tensors()

    # TODO: import below constants from autoencoders/embedders:
    USER = "user"
    SERVICE = "service"

    USER_ONE_HOT_DIM = len(User.objects.first().one_hot_tensor)

    user_autoencoder_mock = AutoEncoder(USER_ONE_HOT_DIM, pipeline_config[UE])
    user_embedder = Embedder(user_autoencoder_mock)

    SERVICE_ONE_HOT_DIM = len(Service.objects.first().one_hot_tensor)

    service_autoencoder_mock = AutoEncoder(SERVICE_ONE_HOT_DIM, pipeline_config[SE])
    service_embedder = Embedder(service_autoencoder_mock)

    user_embedder.save(USER)
    service_embedder.save(SERVICE)


@pytest.fixture
def mock_ncf_pipeline_exec(pipeline_config, mock_autoencoders_pipeline_exec):
    training_step_config = pipeline_config[ModelTrainingStep.__name__]

    users_max_id = User.objects.order_by("-id").first().id
    services_max_id = Service.objects.order_by("-id").first().id

    # TODO: import below constants from autoencoders/embedders:
    USER = "user"
    SERVICE = "service"

    Embedder.load(USER)(User.objects, use_cache=False, save_cache=True)
    Embedder.load(SERVICE)(Service.objects, use_cache=False, save_cache=True)

    model = NeuralCollaborativeFilteringModel(
        users_max_id=users_max_id,
        services_max_id=services_max_id,
        mf_embedding_dim=training_step_config[MF_EMBEDDING_DIM],
        user_ids_embedding_dim=training_step_config[USER_IDS_EMBEDDING_DIM],
        service_ids_embedding_dim=training_step_config[SERVICE_IDS_EMBEDDING_DIM],
        user_emb_dim=pipeline_config[UE],
        service_emb_dim=pipeline_config[SE],
        mlp_layers_spec=training_step_config[MLP_LAYERS_SPEC],
        content_mlp_layers_spec=training_step_config[CONTENT_MLP_LAYERS_SPEC],
    )

    model.save(version=NEURAL_CF)
