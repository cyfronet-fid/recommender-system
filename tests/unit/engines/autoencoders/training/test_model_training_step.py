# pylint: disable-all

import pytest
from random import randint, uniform

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.utils.data import DataLoader

from recommender.engines.autoencoders.training.model_training_step import (
    AEModelTrainingStep,
    ModelTrainingStep,
    ENCODER_LAYER_SIZES,
    DECODER_LAYER_SIZES,
    LOSS,
    LR,
    EPOCHS,
    TRAINING_TIME,
    USER_EMBEDDER,
    SERVICE_EMBEDDER,
    USER_EMBEDDING_DIM,
    SERVICE_EMBEDDING_DIM,
    DATASET,
    EMBEDDER,
    MODEL,
    perform_training,
    create_autoencoder_model,
    autoencoder_loss_function,
    evaluate_autoencoder,
    train_autoencoder,
    get_train_and_valid_datasets,
)
from recommender.engines.autoencoders.training.data_preparation_step import TRAIN, VALID
from recommender.engines.autoencoders.training.data_extraction_step import (
    AUTOENCODERS,
    USERS,
    SERVICES,
)
from recommender.engines.autoencoders.ml_components.embedder import (
    Embedder,
    AutoEncoder,
)
from recommender.engines.constants import DEVICE, WRITER, VERBOSE
from tests.unit.engines.autoencoders.conftest import (
    USER_FEATURES_DIM,
    SERVICE_FEATURES_DIM,
)


def test_model_training_step(simulate_data_preparation_step, ae_pipeline_config):
    """
    Testing:
    -> configuration
    -> training
    -> save embedder
    -> load embedder
    """
    ae_model_train_step = AEModelTrainingStep(ae_pipeline_config)
    # Check the correctness of the configuration inside model training step
    assert ae_pipeline_config[ModelTrainingStep.__name__] == ae_model_train_step.config

    data_prep_step, _ = simulate_data_preparation_step

    data_model_train_step, details_model_train_step = ae_model_train_step(
        data_prep_step
    )

    # Data
    for collection in (USERS, SERVICES):
        # Check the correctness of the returned datasets on which models were trained
        assert (
            data_model_train_step[collection][DATASET]
            is data_prep_step[AUTOENCODERS][collection]
        )

        # Check Embedder class memberships
        assert isinstance(data_model_train_step[collection][EMBEDDER], Embedder)

        # Check AutoEncoder class memberships
        assert isinstance(data_model_train_step[collection][MODEL], AutoEncoder)

    # Details
    for collection in (USERS, SERVICES):
        assert isinstance(details_model_train_step[collection][LOSS], float)
        assert isinstance(details_model_train_step[collection][TRAINING_TIME], float)

    # Save Embedders
    ae_model_train_step.save()
    # Load Embedders
    for embedder in (USER_EMBEDDER, SERVICE_EMBEDDER):
        assert isinstance(Embedder.load(version=embedder), Embedder)


def test_perform_training(training_data):
    """
    Testing:
    -> The whole process of autoencoders training which consists of:
    - creating autoencoder model,
    - training,
    - evaluating.
    """
    data = training_data

    for collection in (USERS, SERVICES):
        if collection == USERS:
            embedding_dim = data[USER_EMBEDDING_DIM]
            features_dim = data[USER_FEATURES_DIM]
        else:
            embedding_dim = data[SERVICE_EMBEDDING_DIM]
            features_dim = data[SERVICE_FEATURES_DIM]

        model, loss, timer = perform_training(
            collection_name=collection,
            train_ds_dl=data[collection][TRAIN],
            valid_ds_dl=data[collection][VALID],
            encoder_layer_sizes=data[ENCODER_LAYER_SIZES],
            decoder_layer_sizes=data[DECODER_LAYER_SIZES],
            embedding_dim=embedding_dim,
            features_dim=features_dim,
            writer=data[WRITER],
            device=data[DEVICE],
            learning_rate=data[LR],
            epochs=data[EPOCHS],
            verbose=data[VERBOSE],
        )

        assert isinstance(model, AutoEncoder)
        assert isinstance(loss, float)
        assert isinstance(timer, float)
        assert timer > 0


def test_create_autoencoder_model(training_data):
    """Testing creating user/service autoencoder model with/without SummaryWriter"""
    data = training_data

    for writer in (None, SummaryWriter()):
        # User
        user_model = create_autoencoder_model(
            collection_name=USERS,
            encoder_layer_sizes=data[ENCODER_LAYER_SIZES],
            decoder_layer_sizes=data[DECODER_LAYER_SIZES],
            features_dim=data[USER_FEATURES_DIM],
            embedding_dim=data[USER_EMBEDDING_DIM],
            writer=writer,
            train_ds_dl=data[USERS][TRAIN],
            device=data[DEVICE],
        )
        assert isinstance(user_model, AutoEncoder)

        # Service
        service_model = create_autoencoder_model(
            collection_name=SERVICES,
            encoder_layer_sizes=data[ENCODER_LAYER_SIZES],
            decoder_layer_sizes=data[DECODER_LAYER_SIZES],
            features_dim=data[SERVICE_FEATURES_DIM],
            embedding_dim=data[SERVICE_EMBEDDING_DIM],
            writer=writer,
            train_ds_dl=data[SERVICES][TRAIN],
            device=data[DEVICE],
        )
        assert isinstance(service_model, AutoEncoder)


def test_train_autoencoder(training_data, get_autoencoder_models):
    """Testing training user/service autoencoder"""
    data = training_data
    ae_models = get_autoencoder_models

    for col, ae_model in ae_models.items():
        optimizer = Adam(ae_model.parameters(), lr=data[LR])
        train_ds_dl = data[USERS][TRAIN] if col == USERS else data[SERVICES][TRAIN]
        valid_ds_dl = data[USERS][VALID] if col == USERS else data[SERVICES][VALID]

        trained_ae_model, timer = train_autoencoder(
            model=ae_model,
            optimizer=optimizer,
            loss_function=autoencoder_loss_function,
            epochs=data[EPOCHS],
            train_ds_dl=train_ds_dl,
            valid_ds_dl=valid_ds_dl,
            writer=data[WRITER],
            save_period=randint(1, 1000),
            verbose=data[VERBOSE],
            device=data[DEVICE],
        )

        assert isinstance(trained_ae_model, AutoEncoder)
        assert isinstance(timer, float)
        assert timer > 0


def test_autoencoder_loss_function(training_data):
    """Testing autoencoders loss function"""
    shape = (randint(2, 1000), randint(2, 1000))

    features = torch.randint(
        high=2, size=shape, dtype=torch.float, device=training_data[DEVICE]
    )
    reconstructions = 6 * torch.rand(shape, requires_grad=True)

    loss = autoencoder_loss_function(reconstructions, features)

    assert loss.is_floating_point()
    assert (
        not loss.is_leaf
    )  # tensors that have requires_grad=False will be leaf tensors by convention
    assert loss.detach().numpy().size == 1


def test_evaluate_autoencoder(training_data, get_autoencoder_models):
    """Testing evaluation of autoencoders"""
    data = training_data
    ae_models = get_autoencoder_models
    device = training_data[DEVICE]

    for col, ae_model in ae_models.items():
        valid_ds_dl = data[USERS][VALID] if col == USERS else data[SERVICES][VALID]
        ae_model = ae_model.to(device)

        val_loss = evaluate_autoencoder(
            model=ae_model,
            dataloader=valid_ds_dl,
            loss_function=autoencoder_loss_function,
            device=device,
        )
        assert isinstance(val_loss, float)


def test_get_train_and_valid_datasets(simulate_data_preparation_step):
    """Testing pytorch's DataLoader on train/valid users/services datasets"""

    data = simulate_data_preparation_step[0][AUTOENCODERS]

    training_datasets = get_train_and_valid_datasets(
        datasets=data,
        user_batch_size=randint(2, 1024),
        service_batch_size=randint(2, 1024),
    )

    for collection in training_datasets.values():
        assert TRAIN and VALID in collection
        assert len(collection) == 2

    # Check class memberships
    for collection in (USERS, SERVICES):
        for split in (TRAIN, VALID):
            assert isinstance(training_datasets[collection][split], DataLoader)
