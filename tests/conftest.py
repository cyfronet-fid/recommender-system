# pylint: disable-all

"""Fixtures used by pytest shared across all tests"""
from random import seed, randint

import pytest
import mongoengine
import torch
from torch.nn import CosineEmbeddingLoss, BCELoss
from torch.optim import Adam
from pymongo import uri_parser

from recommender import create_app, User
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
    TRAIN_DS_SIZE,
    VALID_DS_SIZE,
    precalc_users_and_service_tensors,
)
from recommender.engines.autoencoders.training.data_validation_step import (
    LEAST_NUM_OF_USR_SRV,
)
from recommender.engines.autoencoders.training.model_evaluation_step import BATCH_SIZE
from recommender.engines.autoencoders.training.model_training_step import (
    LOSS_FUNCTION,
    USER_BATCH_SIZE,
    SERVICE_BATCH_SIZE,
    USER_EMBEDDING_DIM,
    SERVICE_EMBEDDING_DIM,
    EPOCHS,
    OPTIMIZER,
    LR,
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
from recommender.engines.ncf.ml_components.neural_collaborative_filtering import (
    NeuralCollaborativeFilteringModel,
    NEURAL_CF,
)
from recommender.engines.ncf.training.data_extraction_step import MAX_USERS
from recommender.engines.ncf.training.data_preparation_step import (
    TRAIN_DS_SIZE,
    VALID_DS_SIZE,
)
from recommender.engines.ncf.training.data_validation_step import (
    LEAST_N_ORDERS_PER_USER,
)
from recommender.engines.ncf.training.model_training_step import (
    BATCH_SIZE,
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
    MAX_ITEMSPACE_SIZE,
    MIN_WEIGHTED_AVG_F1_SCORE,
)
from recommender.engines.rl.ml_components.synthetic_dataset.rewards import (
    RewardGeneration,
)
from recommender.engines.rl.training.data_extraction_step.data_extraction_step import (
    K,
    SYNTHETIC_PARAMS,
    INTERACTIONS_RANGE,
    REWARD_GENERATION_MODE,
    MIN_USER_ACTIONS,
    MIN_RECOMMENDATIONS,
    GENERATE_NEW,
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
    LEARNING_RATE,
    CRITIC_OPTIMIZER,
    CRITIC_OPTIMIZER_PARAMS,
    TARGET_NOISE,
    NOISE_CLIP,
    GAMMA,
    POLICY_DELAY,
    ACT_MIN,
    ACT_MAX,
    RL_EPOCHS,
)
from recommender.engines.rl.training.model_validation_step.model_validation_step import (
    TIME_UPPER_BOUND,
    REWARD_LOWER_BOUND,
)
from recommender.extensions import db
from recommender.models import Service
from tests.factories.populate_database import populate_users_and_services


@pytest.fixture()
def _app():
    return create_app()


@pytest.fixture
def client(_app):
    """Flask app client that you can make HTTP requests to"""
    yield _app.test_client()
    mongoengine.connection.disconnect_all()


@pytest.fixture
def mongo(_app):
    """MongoDB fixture"""
    with _app.app_context():
        yield db
        uri = db.app.config["MONGODB_HOST"]
        db_name = uri_parser.parse_uri(uri)["database"]
        db.connection.drop_database(db_name)
        mongoengine.connection.disconnect_all()


@pytest.fixture
def ae_pipeline_config(embedding_dims):
    user_embedding_dim, service_embedding_dim = embedding_dims
    config = {
        DEVICE: torch.device("cpu"),
        WRITER: None,
        VERBOSE: True,
        LOSS_FUNCTION: CosineEmbeddingLoss(),
        DataExtractionStep.__name__: {},
        DataValidationStep.__name__: {LEAST_NUM_OF_USR_SRV: 2},
        DataPreparationStep.__name__: {TRAIN_DS_SIZE: 0.6, VALID_DS_SIZE: 0.2},
        ModelTrainingStep.__name__: {
            USER_BATCH_SIZE: 128,
            SERVICE_BATCH_SIZE: 128,
            USER_EMBEDDING_DIM: user_embedding_dim,
            SERVICE_EMBEDDING_DIM: service_embedding_dim,
            EPOCHS: 500,
            OPTIMIZER: Adam,
            LR: 0.01,
        },
        ModelEvaluationStep.__name__: {BATCH_SIZE: 128},
        ModelValidationStep.__name__: {MAX_LOSS_SCORE: 1.5},
    }

    return config


@pytest.fixture
def ncf_pipeline_config(embedding_dims):
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


def users_services_args(valid=True):
    seed(10)
    args = {
        "common_services_num": randint(10, 20),
        "unordered_services_num": randint(10, 20),
        "users_num": randint(1, 12),
        "k_common_services_min": randint(1, 3),
        "k_common_services_max": randint(4, 6),
        "verbose": False,
        "valid": valid,
    }

    return args


@pytest.fixture
def generate_users_and_services(mongo):
    args = users_services_args(valid=True)
    populate_users_and_services(**args)


@pytest.fixture
def generate_invalid_users_and_services(mongo):
    args = users_services_args(valid=False)
    populate_users_and_services(**args)


@pytest.fixture
def delete_users_services():
    User.drop_collection()
    Service.drop_collection()


@pytest.fixture
def embedding_dims():
    user_embedding_dim = 64
    service_embedding_dim = 128
    return user_embedding_dim, service_embedding_dim


@pytest.fixture
def embedding_exec(mock_autoencoders_pipeline_exec):
    embedding_component = EmbeddingComponent()
    embedding_component()


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
def mock_ncf_pipeline_exec(ncf_pipeline_config, mock_autoencoders_pipeline_exec):
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


@pytest.fixture
def mp_dump_data():
    """Example MP database dump"""
    return {
        "services": [
            {
                "id": 1,
                "name": "test",
                "description": "desc",
                "tagline": "tag",
                "countries": ["PL", "US"],
                "rating": "2",
                "order_type": "open_access",
                "categories": [1, 2],
                "providers": [1, 2],
                "resource_organisation": 1,
                "scientific_domains": [1, 2],
                "platforms": [1, 2],
                "target_users": [1, 2],
                "related_services": [2],
                "required_services": [2],
                "access_types": [1, 2],
                "access_modes": [1, 2],
                "trls": [1, 2],
                "life_cycle_statuses": [1, 2],
            },
            {
                "id": 2,
                "name": "test2",
                "description": "desc",
                "tagline": "tag",
                "countries": ["PL"],
                "rating": "2",
                "order_type": "open_access",
                "categories": [2],
                "providers": [2],
                "resource_organisation": 2,
                "scientific_domains": [2],
                "platforms": [2],
                "target_users": [2],
                "related_services": [1],
                "required_services": [],
                "access_types": [2],
                "access_modes": [2],
                "trls": [2],
                "life_cycle_statuses": [2],
            },
        ],
        "users": [
            {
                "id": 1,
                "scientific_domains": [1, 2],
                "categories": [1, 2],
                "accessed_services": [1, 2],
            }
        ],
        "categories": [{"id": 1, "name": "c1"}, {"id": 2, "name": "c2"}],
        "providers": [{"id": 1, "name": "p1"}, {"id": 2, "name": "p2"}],
        "scientific_domains": [{"id": 1, "name": "sd1"}, {"id": 2, "name": "sd2"}],
        "platforms": [{"id": 1, "name": "pl1"}, {"id": 2, "name": "pl2"}],
        "target_users": [
            {"id": 1, "name": "tu1", "description": "desc"},
            {"id": 2, "name": "tu2", "description": "desc"},
        ],
        "access_modes": [
            {"id": 1, "name": "am1", "description": "desc"},
            {"id": 2, "name": "am2", "description": "desc"},
        ],
        "access_types": [
            {"id": 1, "name": "at1", "description": "desc"},
            {"id": 2, "name": "at2", "description": "desc"},
        ],
        "trls": [
            {"id": 1, "name": "trl-1", "description": "desc"},
            {"id": 2, "name": "trl-2", "description": "desc"},
        ],
        "life_cycle_statuses": [
            {"id": 1, "name": "lcs1", "description": "desc"},
            {"id": 2, "name": "lcs2", "description": "desc"},
        ],
    }


@pytest.fixture
def recommendation_json_dict():
    """Fixture of json dict of the recommendations endpoint request"""

    return {
        "user_id": 1,
        "unique_id": "5642c351-80fe-44cf-b606-304f2f338122",
        "timestamp": "2021-03-25T12:43:53.118Z",
        "visit_id": "202090a4-de4c-4230-acba-6e2931d9e37c",
        "page_id": "some_page_identifier",
        "panel_id": "v1",
        "services": [1, 2, 3],
        "search_data": {
            "q": "Cloud GPU",
            "categories": [1],
            "geographical_availabilities": ["PL"],
            "order_type": "open_access",
            "providers": [1],
            "rating": "5",
            "related_platforms": [1],
            "scientific_domains": [1],
            "sort": "_score",
            "target_users": [1],
        },
    }


@pytest.fixture
def user_action_json_dict():
    """Fixture of json dict of the user_actions endpoint request"""
    return {
        "user_id": 1,
        "unique_id": "5642c351-80fe-44cf-b606-304f2f338122",
        "timestamp": "2021-03-25T14:10:42.368Z",
        "source": {
            "visit_id": "202090a4-de4c-4230-acba-6e2931d9e37c",
            "page_id": "services_catalogue_list",
            "root": {"type": "recommendation_panel", "panel_id": "v1", "service_id": 1},
        },
        "target": {
            "visit_id": "9f543b80-dd5b-409b-a619-6312a0b04f4f",
            "page_id": "service_about",
        },
        "action": {"type": "button", "text": "Details", "order": True},
    }
