# pylint: disable-all

import pytest
import torch

from recommender.engines.autoencoders.training.model_training_step import (
    SERVICE_EMBEDDING_DIM,
    USER_EMBEDDING_DIM,
    LR,
)
from recommender.engines.base.base_steps import (
    DataExtractionStep,
    DataValidationStep,
    DataPreparationStep,
    ModelTrainingStep,
    ModelEvaluationStep,
    ModelValidationStep,
)
from recommender.engines.constants import VERBOSE, DEVICE
from recommender.engines.rl.ml_components.synthetic_dataset.rewards import (
    RewardGeneration,
)
from recommender.engines.rl.training.data_extraction_step.data_extraction_step import (
    GENERATE_NEW,
    MIN_RECOMMENDATIONS,
    MIN_USER_ACTIONS,
    K,
    SYNTHETIC_PARAMS,
    INTERACTIONS_RANGE,
    REWARD_GENERATION_MODE,
)
from recommender.engines.rl.training.data_preparation_step.data_preparation_step import (
    SHUFFLE,
    SARS_BATCH_SIZE,
)
from recommender.engines.rl.training.data_validation_step.data_validation_step import (
    MIN_EMPTY_TO_NON_EMPTY_RATIO,
    MIN_SARSES,
)
from recommender.engines.rl.training.model_evaluation_step.model_evaluation_step import (
    TIME_MEASUREMENT_SAMPLES,
)
from recommender.engines.rl.training.model_training_step.model_training_step import (
    RL_EPOCHS,
    ACT_MAX,
    ACT_MIN,
    POLICY_DELAY,
    GAMMA,
    NOISE_CLIP,
    TARGET_NOISE,
    CRITIC_OPTIMIZER_PARAMS,
    CRITIC_OPTIMIZER,
    ACTOR_OPTIMIZER_PARAMS,
    ACTOR_OPTIMIZER,
    CRITIC_LAYER_SIZES,
    ACTOR_LAYER_SIZES,
    POLYAK,
    HISTORY_LEN,
    LEARNING_RATE,
)
from recommender.engines.rl.training.model_validation_step.model_validation_step import (
    TIME_UPPER_BOUND,
    REWARD_LOWER_BOUND,
)


@pytest.fixture
def rl_pipeline_config(embedding_dims):
    user_embedding_dim, service_embedding_dim = embedding_dims
    config = {
        SERVICE_EMBEDDING_DIM: service_embedding_dim,
        USER_EMBEDDING_DIM: user_embedding_dim,
        DEVICE: "cpu",
        VERBOSE: True,
        K: 3,
        SYNTHETIC_PARAMS: {
            K: 3,
            INTERACTIONS_RANGE: (1, 2),
            REWARD_GENERATION_MODE: RewardGeneration.COMPLEX,
        },
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
