# pylint: disable-all
"""Fixtures for ncf"""

import pytest
from typing import Tuple, Dict
import torch
from torch.nn import BCELoss
from torch.optim import Adam

from recommender import User
from recommender.engines.autoencoders.ml_components.embedder import (
    Embedder,
    USER_EMBEDDER,
    SERVICE_EMBEDDER,
)
from recommender.engines.autoencoders.training.data_preparation_step import (
    TRAIN_DS_SIZE,
    VALID_DS_SIZE,
)
from recommender.engines.autoencoders.training.model_evaluation_step import BATCH_SIZE
from recommender.engines.autoencoders.training.model_training_step import (
    USER_EMBEDDING_DIM,
    SERVICE_EMBEDDING_DIM,
    LOSS_FUNCTION,
    EPOCHS,
    OPTIMIZER,
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
from recommender.engines.ncf.ml_components.neural_collaborative_filtering import (
    NeuralCollaborativeFilteringModel,
    NEURAL_CF,
)
from recommender.engines.ncf.training.data_extraction_step import MAX_USERS
from recommender.engines.ncf.training.data_validation_step import (
    LEAST_N_ORDERS_PER_USER,
)
from recommender.engines.ncf.training.model_training_step import (
    MF_EMBEDDING_DIM,
    USER_IDS_EMBEDDING_DIM,
    SERVICE_IDS_EMBEDDING_DIM,
    MLP_LAYERS_SPEC,
    CONTENT_MLP_LAYERS_SPEC,
    OPTIMIZER_PARAMS,
)
from recommender.engines.ncf.training.model_validation_step import (
    MAX_EXECUTION_TIME,
    MAX_ITEMSPACE_SIZE,
    MIN_WEIGHTED_AVG_F1_SCORE,
)
from recommender.models import Service


@pytest.fixture
def ncf_pipeline_config(embedding_dims: Tuple[int, int]) -> Dict:
    """NCF pipline configuration"""
    user_embedding_dim, service_embedding_dim = embedding_dims
    config = {
        DEVICE: torch.device("cpu"),
        WRITER: None,
        VERBOSE: False,
        BATCH_SIZE: 64,
        SERVICE_EMBEDDING_DIM: service_embedding_dim,
        USER_EMBEDDING_DIM: user_embedding_dim,
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
            # remote github actions CI pipeline. It should be ofc about 0.1 [s]
            MAX_EXECUTION_TIME: 500,
            MAX_ITEMSPACE_SIZE: 1000,
            MIN_WEIGHTED_AVG_F1_SCORE: 0.05,
        },
    }

    return config


@pytest.fixture
def mock_ncf_pipeline_exec(ncf_pipeline_config, mock_autoencoders_pipeline_exec):
    """Mock execution of NCF pipline"""
    training_step_config = ncf_pipeline_config[ModelTrainingStep.__name__]

    users_max_id = User.objects.order_by("-id").first().id
    services_max_id = Service.objects.order_by("-id").first().id

    Embedder.load(USER_EMBEDDER)(User.objects, use_cache=False, save_cache=True)
    Embedder.load(SERVICE_EMBEDDER)(Service.objects, use_cache=False, save_cache=True)

    model = NeuralCollaborativeFilteringModel(
        users_max_id=users_max_id,
        services_max_id=services_max_id,
        mf_embedding_dim=training_step_config[MF_EMBEDDING_DIM],
        user_ids_embedding_dim=training_step_config[USER_IDS_EMBEDDING_DIM],
        service_ids_embedding_dim=training_step_config[SERVICE_IDS_EMBEDDING_DIM],
        user_emb_dim=ncf_pipeline_config[USER_EMBEDDING_DIM],
        service_emb_dim=ncf_pipeline_config[SERVICE_EMBEDDING_DIM],
        mlp_layers_spec=training_step_config[MLP_LAYERS_SPEC],
        content_mlp_layers_spec=training_step_config[CONTENT_MLP_LAYERS_SPEC],
    )

    model.save(version=NEURAL_CF)
