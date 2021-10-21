# pylint: disable-all

import pytest
import torch

from recommender.engine.models.autoencoders import AutoEncoder
from recommender.engines.autoencoders.ml_components.embedder import Embedder
from recommender.engines.autoencoders.ml_components.normalizer import Normalizer
from recommender.engines.autoencoders.training.data_preparation_step import (
    precalc_users_and_service_tensors,
)
from recommender.engines.base.base_steps import (
    DataExtractionStep,
    DataValidationStep,
    DataPreparationStep,
    ModelTrainingStep,
    ModelEvaluationStep,
    ModelValidationStep,
)
from recommender.engines.rl.training.pipeline import RLPipeline
from recommender.models import User, Service
from recommender.services.synthetic_dataset.rewards import RewardGeneration
from tests.factories.populate_database import populate_users_and_services


# TODO: Merge fixtures with other pipelines
@pytest.fixture
def generate_data(mongo):
    populate_users_and_services(
        common_services_number=9,
        no_one_services_number=9,
        users_number=5,
        k_common_services_min=5,
        k_common_services_max=7,
    )


@pytest.fixture
def pipeline_config():
    return {
        "K": 3,
        DataExtractionStep.__name__: {
            "synthetic": True,
            "generate_new": True,
            "synthetic_params": {
                "interactions_range": (1, 2),
                "reward_generation_mode": RewardGeneration.SIMPLE,
                "simple_reward_threshold": 0.5,
            },
        },
        DataValidationStep.__name__: {"minimum_sarses": 0},
        DataPreparationStep.__name__: {"batch_size": 32, "shuffle": True},
        ModelTrainingStep.__name__: {
            "SE": 32,
            "UE": 64,
            "N": 20,
            "device": "cpu",
            "polyak": 0.95,
            "actor_layer_sizes": (128, 256, 128),
            "critic_layer_sizes": (128, 256, 128),
            "actor_optimizer": torch.optim.Adam,
            "actor_optimizer_params": {"lr": 1e-3},
            "critic_optimizer": torch.optim.Adam,
            "critic_optimizer_params": {"lr": 1e-4},
            "target_noise": 0.4,
            "noise_clip": 0.5,
            "gamma": 1.0,
            "policy_delay": 2,
            "act_min": -1.0,
            "act_max": 1.0,
            "epochs": 10,
        },
        ModelEvaluationStep.__name__: {"time_measurement_samples": 50},
        ModelValidationStep.__name__: {
            "time_upper_bound": 100.0,
            "reward_lower_bound": 0,
        },
    }


@pytest.fixture
def mock_autoencoders_pipeline_exec(pipeline_config):
    precalc_users_and_service_tensors()

    # TODO: import below constants from autoencoders/embedders:
    USER = "user"
    SERVICE = "service"

    USER_ONE_HOT_DIM = len(User.objects.first().one_hot_tensor)

    user_autoencoder_mock = AutoEncoder(
        USER_ONE_HOT_DIM, pipeline_config[ModelTrainingStep.__name__]["UE"]
    )
    user_embedder = Embedder(user_autoencoder_mock)

    SERVICE_ONE_HOT_DIM = len(Service.objects.first().one_hot_tensor)

    service_autoencoder_mock = AutoEncoder(
        SERVICE_ONE_HOT_DIM, pipeline_config[ModelTrainingStep.__name__]["SE"]
    )
    service_embedder = Embedder(service_autoencoder_mock)

    user_embedder.save(USER)
    service_embedder.save(SERVICE)


def test_rl_pipeline(
    mongo, generate_data, pipeline_config, mock_autoencoders_pipeline_exec
):
    # TODO: Extract dense tensor caching outside of pipelines
    user_embedder = Embedder.load(version="user")
    service_embedder = Embedder.load(version="service")
    user_embedder(User.objects, use_cache=False, save_cache=True)
    service_embedder(Service.objects, use_cache=False, save_cache=True)
    normalizer = Normalizer()
    normalizer(User.objects, save_cache=True)
    normalizer(Service.objects, save_cache=True)

    rl_pipeline = RLPipeline(pipeline_config)
    rl_pipeline()
