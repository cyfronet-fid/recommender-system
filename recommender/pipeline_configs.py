# pylint: disable=fixme, line-too-long

"""This file contains all pipelines' configs"""

import torch
from torch.nn import BCELoss, CosineEmbeddingLoss
from torch.optim import Adam

from recommender.engines.autoencoders.training.data_validation_step import (
    LEAST_NUM_OF_USR_SRV,
)
from recommender.engines.autoencoders.training.model_evaluation_step import (
    BATCH_SIZE as AE_BATCH_SIZE,
)
from recommender.engines.autoencoders.training.model_validation_step import (
    MAX_LOSS_SCORE,
)
from recommender.engines.base.base_steps import (
    DataExtractionStep,
    DataValidationStep,
    DataPreparationStep,
    ModelTrainingStep,
    ModelEvaluationStep,
    ModelValidationStep,
)
from recommender.engines.constants import DEVICE, WRITER, VERBOSE
from recommender.engines.ncf.training.data_extraction_step import MAX_USERS
from recommender.engines.ncf.training.data_preparation_step import (
    TRAIN_DS_SIZE as NCF_TRAIN_DS_SIZE,
    VALID_DS_SIZE as NCF_VALID_DS_SIZE,
)
from recommender.engines.autoencoders.training.data_preparation_step import (
    TRAIN_DS_SIZE,
    VALID_DS_SIZE,
)

from recommender.engines.ncf.training.data_validation_step import (
    LEAST_N_ORDERS_PER_USER,
)
from recommender.engines.ncf.training.model_training_step import (
    MF_EMBEDDING_DIM,
    USER_IDS_EMBEDDING_DIM,
    SERVICE_IDS_EMBEDDING_DIM,
    MLP_LAYERS_SPEC,
    CONTENT_MLP_LAYERS_SPEC,
)

from recommender.engines.ncf.training.model_training_step import (
    BATCH_SIZE as NCF_BATCH_SIZE,
)

from recommender.engines.ncf.training.model_training_step import EPOCHS as NCF_EPOCHS
from recommender.engines.autoencoders.training.model_training_step import (
    EPOCHS as AE_EPOCHS,
    LR,
)
from recommender.engines.autoencoders.training.model_training_step import (
    OPTIMIZER as AE_OPTIMIZER,
)

from recommender.engines.ncf.training.model_training_step import (
    OPTIMIZER as NCF_OPTIMIZER,
    OPTIMIZER_PARAMS as NCF_OPTIMIZER_PARAMS,
)

from recommender.engines.ncf.training.model_validation_step import (
    MAX_EXECUTION_TIME,
    MAX_ITEMSPACE_SIZE,
    MIN_WEIGHTED_AVG_F1_SCORE,
)

from recommender.engines.autoencoders.training.model_training_step import (
    LOSS_FUNCTION as AE_LOSS_FUNCTION,
    ENCODER_LAYER_SIZES,
    DECODER_LAYER_SIZES,
    USER_BATCH_SIZE,
    SERVICE_BATCH_SIZE,
    USER_EMBEDDING_DIM,
    SERVICE_EMBEDDING_DIM,
)
from recommender.engines.ncf.training.model_training_step import (
    LOSS_FUNCTION as NCF_LOSS_FUNCTION,
)
from recommender.engines.rl.ml_components.synthetic_dataset.rewards import (
    RewardGeneration,
)
from recommender.engines.rl.training.data_extraction_step.data_extraction_step import (
    MIN_USER_ACTIONS,
    MIN_RECOMMENDATIONS,
    SYNTHETIC_PARAMS,
    K,
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
    POLYAK,
    ACTOR_LAYER_SIZES,
    CRITIC_LAYER_SIZES,
    ACTOR_OPTIMIZER,
    ACTOR_OPTIMIZER_PARAMS,
    CRITIC_OPTIMIZER,
    CRITIC_OPTIMIZER_PARAMS,
    TARGET_NOISE,
    NOISE_CLIP,
    GAMMA,
    POLICY_DELAY,
    ACT_MIN,
    ACT_MAX,
    RL_EPOCHS,
    LEARNING_RATE,
)
from recommender.engines.rl.training.model_validation_step.model_validation_step import (
    TIME_UPPER_BOUND,
    REWARD_LOWER_BOUND,
)

SE = 128
UE = 64
COMPUTING_DEVICE = torch.device("cpu")

AUTOENCODERS_PIPELINE_CONFIG = {
    DEVICE: COMPUTING_DEVICE,
    WRITER: None,
    VERBOSE: True,
    AE_LOSS_FUNCTION: CosineEmbeddingLoss(),
    DataExtractionStep.__name__: {},
    DataValidationStep.__name__: {LEAST_NUM_OF_USR_SRV: 2},
    DataPreparationStep.__name__: {TRAIN_DS_SIZE: 0.96, VALID_DS_SIZE: 0.02},
    ModelTrainingStep.__name__: {
        ENCODER_LAYER_SIZES: (128, 64),
        DECODER_LAYER_SIZES: (64, 128),
        USER_BATCH_SIZE: 128,
        SERVICE_BATCH_SIZE: 128,
        SERVICE_EMBEDDING_DIM: SE,
        USER_EMBEDDING_DIM: UE,
        AE_EPOCHS: 200,
        AE_OPTIMIZER: Adam,
        LR: 0.01,
    },
    ModelEvaluationStep.__name__: {AE_BATCH_SIZE: 128},
    ModelValidationStep.__name__: {MAX_LOSS_SCORE: 2},
}

NCF_PIPELINE_CONFIG = {
    DEVICE: COMPUTING_DEVICE,
    WRITER: None,
    VERBOSE: True,
    NCF_BATCH_SIZE: 64,
    SERVICE_EMBEDDING_DIM: SE,
    USER_EMBEDDING_DIM: UE,
    NCF_LOSS_FUNCTION: BCELoss(),
    DataExtractionStep.__name__: {MAX_USERS: None},
    DataValidationStep.__name__: {LEAST_N_ORDERS_PER_USER: 5},
    DataPreparationStep.__name__: {NCF_TRAIN_DS_SIZE: 0.6, NCF_VALID_DS_SIZE: 0.2},
    ModelTrainingStep.__name__: {
        MF_EMBEDDING_DIM: 64,
        USER_IDS_EMBEDDING_DIM: 64,
        SERVICE_IDS_EMBEDDING_DIM: 64,
        MLP_LAYERS_SPEC: (64, 32, 16, 8),
        CONTENT_MLP_LAYERS_SPEC: (128, 64, 32),
        NCF_EPOCHS: 100,
        NCF_OPTIMIZER: Adam,
        NCF_OPTIMIZER_PARAMS: {"lr": 0.01},
    },
    ModelEvaluationStep.__name__: {},
    ModelValidationStep.__name__: {
        # TODO: Below MAX_EXECUTION_TIME is so high and MAX_ITEMSPACE_SIZE
        #  is low because pipeline shouldn' fail for now
        MAX_EXECUTION_TIME: 100,
        MAX_ITEMSPACE_SIZE: 1000,
        MIN_WEIGHTED_AVG_F1_SCORE: 0.1,
    },
}

RL_PIPELINE_CONFIG_BASE = {
    SERVICE_EMBEDDING_DIM: SE,
    USER_EMBEDDING_DIM: UE,
    DEVICE: COMPUTING_DEVICE,
    VERBOSE: True,
    DataExtractionStep.__name__: {
        MIN_USER_ACTIONS: 2500,
        MIN_RECOMMENDATIONS: 2500,
    },
    DataValidationStep.__name__: {
        MIN_SARSES: 0,
        MIN_EMPTY_TO_NON_EMPTY_RATIO: float("inf"),
    },
    DataPreparationStep.__name__: {SARS_BATCH_SIZE: 64, SHUFFLE: True},
    ModelTrainingStep.__name__: {
        POLYAK: 0.95,
        ACTOR_LAYER_SIZES: (128, 256, 128),
        CRITIC_LAYER_SIZES: (128, 256, 128),
        HISTORY_LEN: 5,
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
        RL_EPOCHS: 100,
    },
    ModelEvaluationStep.__name__: {TIME_MEASUREMENT_SAMPLES: 50},
    ModelValidationStep.__name__: {
        TIME_UPPER_BOUND: 100.0,
        REWARD_LOWER_BOUND: 0,
    },
}

RL_PIPELINE_CONFIG = {
    **{
        K: 3,
        SYNTHETIC_PARAMS: {
            K: 3,
            INTERACTIONS_RANGE: (1, 2),
            REWARD_GENERATION_MODE: RewardGeneration.COMPLEX,
        },
    },
    **RL_PIPELINE_CONFIG_BASE,
}
