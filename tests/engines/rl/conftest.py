# pylint: disable-all
"""Fixtures for rl"""
import pytest
from typing import Tuple, Dict
import torch

from recommender.engines.autoencoders.training.model_training_step import (
    SERVICE_EMBEDDING_DIM,
    USER_EMBEDDING_DIM,
)
from recommender.engines.base.base_steps import (
    ModelTrainingStep,
    DataExtractionStep,
    DataValidationStep,
    DataPreparationStep,
    ModelEvaluationStep,
    ModelValidationStep,
)
from recommender.engines.constants import DEVICE, VERBOSE
from recommender.engines.panel_id_to_services_number_mapping import K_TO_PANEL_ID
from recommender.engines.rl.ml_components.actor import Actor
from recommender.engines.rl.ml_components.history_embedder import MLPHistoryEmbedder
from recommender.engines.rl.ml_components.synthetic_dataset.rewards import (
    RewardGeneration,
)
from recommender.engines.rl.training.data_extraction_step.data_extraction_step import (
    K,
    MIN_USER_ACTIONS,
    MIN_RECOMMENDATIONS,
    GENERATE_NEW,
    SYNTHETIC_PARAMS,
    INTERACTIONS_RANGE,
    REWARD_GENERATION_MODE,
)
from recommender.engines.rl.training.data_preparation_step.data_preparation_step import (
    SARS_BATCH_SIZE,
    SHUFFLE,
)
from recommender.engines.rl.training.data_validation_step.data_validation_step import (
    MIN_SARSES,
    MIN_EMPTY_TO_NON_EMPTY_RATIO,
)
from recommender.engines.rl.training.model_evaluation_step.model_evaluation_step import (
    TIME_MEASUREMENT_SAMPLES,
)
from recommender.engines.rl.training.model_training_step.model_training_step import (
    HISTORY_LEN,
    ACTOR_LAYER_SIZES,
    ACT_MAX,
    ACT_MIN,
    POLYAK,
    CRITIC_LAYER_SIZES,
    ACTOR_OPTIMIZER,
    ACTOR_OPTIMIZER_PARAMS,
    LEARNING_RATE,
    CRITIC_OPTIMIZER,
    CRITIC_OPTIMIZER_PARAMS,
    TARGET_NOISE,
    NOISE_CLIP,
    GAMMA,
    POLICY_DELAY,
    RL_EPOCHS,
)
from recommender.engines.rl.training.model_validation_step.model_validation_step import (
    TIME_UPPER_BOUND,
    REWARD_LOWER_BOUND,
)
from recommender.models import Service


@pytest.fixture
def base_rl_pipeline_config(embedding_dims: Tuple[int, int]) -> Dict:
    """Base configuration of RL pipline"""
    user_embedding_dim, service_embedding_dim = embedding_dims
    config = {
        SERVICE_EMBEDDING_DIM: service_embedding_dim,
        USER_EMBEDDING_DIM: user_embedding_dim,
        DEVICE: "cpu",
        VERBOSE: True,
        DataExtractionStep.__name__: {
            MIN_USER_ACTIONS: 2500,
            MIN_RECOMMENDATIONS: 2500,
            GENERATE_NEW: True,
        },
        DataValidationStep.__name__: {
            MIN_SARSES: 0,
            MIN_EMPTY_TO_NON_EMPTY_RATIO: float("inf"),
        },
        DataPreparationStep.__name__: {SARS_BATCH_SIZE: 64, SHUFFLE: True},
        ModelTrainingStep.__name__: {
            HISTORY_LEN: 20,
            POLYAK: 0.95,
            ACTOR_LAYER_SIZES: (128, 256, 128),
            CRITIC_LAYER_SIZES: (128, 256, 128),
            ACTOR_OPTIMIZER: torch.optim.Adam,
            ACTOR_OPTIMIZER_PARAMS: {LEARNING_RATE: 1e-3},
            CRITIC_OPTIMIZER: torch.optim.Adam,
            CRITIC_OPTIMIZER_PARAMS: {LEARNING_RATE: 1e-4},
            TARGET_NOISE: 0.4,
            NOISE_CLIP: 0.5,
            GAMMA: 1.0,
            POLICY_DELAY: 2,
            ACT_MIN: -1.0,
            ACT_MAX: 1.0,
            RL_EPOCHS: 2,
        },
        ModelEvaluationStep.__name__: {TIME_MEASUREMENT_SAMPLES: 50},
        ModelValidationStep.__name__: {
            TIME_UPPER_BOUND: 100.0,
            REWARD_LOWER_BOUND: 0,
        },
    }

    return config


@pytest.fixture()
def rl_pipeline_v1_config(base_rl_pipeline_config: Dict) -> Dict:
    """V1 configuration of RL pipline"""
    return {
        **{
            K: 3,
            SYNTHETIC_PARAMS: {
                K: 3,
                INTERACTIONS_RANGE: (1, 2),
                REWARD_GENERATION_MODE: RewardGeneration.COMPLEX,
            },
        },
        **base_rl_pipeline_config,
    }


@pytest.fixture()
def rl_pipeline_v2_config(base_rl_pipeline_config: Dict) -> Dict:
    """V2 configuration of RL pipline"""
    return {
        **{
            K: 2,
            SYNTHETIC_PARAMS: {
                K: 2,
                INTERACTIONS_RANGE: (1, 2),
                REWARD_GENERATION_MODE: RewardGeneration.COMPLEX,
            },
        },
        **base_rl_pipeline_config,
    }


@pytest.fixture
def mock_rl_pipeline_exec(
    rl_pipeline_v1_config,
    rl_pipeline_v2_config,
    mock_autoencoders_pipeline_exec,
    embedding_exec,
):
    """Mock execution of RL pipline"""
    actor_v1 = Actor(
        K=rl_pipeline_v1_config[K],
        SE=rl_pipeline_v1_config[SERVICE_EMBEDDING_DIM],
        UE=rl_pipeline_v1_config[USER_EMBEDDING_DIM],
        I=len(Service.objects),
        history_embedder=MLPHistoryEmbedder(
            SE=rl_pipeline_v1_config[SERVICE_EMBEDDING_DIM],
            max_N=rl_pipeline_v1_config[ModelTrainingStep.__name__][HISTORY_LEN],
        ),
        layer_sizes=rl_pipeline_v1_config[ModelTrainingStep.__name__][
            ACTOR_LAYER_SIZES
        ],
        act_max=rl_pipeline_v1_config[ModelTrainingStep.__name__][ACT_MAX],
        act_min=rl_pipeline_v1_config[ModelTrainingStep.__name__][ACT_MIN],
    )

    actor_v2 = Actor(
        K=rl_pipeline_v2_config[K],
        SE=rl_pipeline_v2_config[SERVICE_EMBEDDING_DIM],
        UE=rl_pipeline_v2_config[USER_EMBEDDING_DIM],
        I=len(Service.objects),
        history_embedder=MLPHistoryEmbedder(
            SE=rl_pipeline_v2_config[SERVICE_EMBEDDING_DIM],
            max_N=rl_pipeline_v2_config[ModelTrainingStep.__name__][HISTORY_LEN],
        ),
        layer_sizes=rl_pipeline_v2_config[ModelTrainingStep.__name__][
            ACTOR_LAYER_SIZES
        ],
        act_max=rl_pipeline_v2_config[ModelTrainingStep.__name__][ACT_MAX],
        act_min=rl_pipeline_v2_config[ModelTrainingStep.__name__][ACT_MIN],
    )

    actor_v1.save(version=K_TO_PANEL_ID.get(rl_pipeline_v1_config[K]))
    actor_v2.save(version=K_TO_PANEL_ID.get(rl_pipeline_v2_config[K]))
