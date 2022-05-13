# pylint: disable-all
import pytest

from recommender.models.user import User
from recommender.models.service import Service
from recommender.engines.autoencoders.inference.embedding_component import (
    EmbeddingComponent,
)
from recommender.engines.autoencoders.training.model_training_step import (
    AEModelTrainingStep,
)
from tests.engines.autoencoders.conftest import embedding_dims


def test_embedding_component(
    simulate_data_preparation_step, embedding_dims, ae_pipeline_config
):
    """
    Testing:
    -> obtaining proper dense_tensors for Users.objects and Service.objects
    """
    user_embedding_dim, service_embedding_dim = embedding_dims
    ae_model_train_step = AEModelTrainingStep(ae_pipeline_config)

    data_prep_step, _ = simulate_data_preparation_step
    _, _ = ae_model_train_step(data_prep_step)

    # Save Embedders
    ae_model_train_step.save()

    users = list(User.objects)
    services = list(Service.objects)

    for user in users:
        assert len(user.one_hot_tensor) > 0
        assert len(user.dense_tensor) == 0

    for service in services:
        assert len(service.one_hot_tensor) > 0
        assert len(service.dense_tensor) == 0

    EmbeddingComponent()()

    users = list(User.objects)
    services = list(Service.objects)

    for user in users:
        assert len(user.one_hot_tensor) > 0
        assert len(user.dense_tensor) == user_embedding_dim

    for service in services:
        assert len(service.one_hot_tensor) > 0
        assert len(service.dense_tensor) == service_embedding_dim
