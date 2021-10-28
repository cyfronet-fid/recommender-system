# pylint: disable-all
"""Fixtures which enable creating isolated environment for unit testing of the autoencoders pipeline steps"""

import pytest
from random import randint, uniform

from torch.utils.data import DataLoader

from recommender.models import User, Service
from recommender.engines.autoencoders.training.data_extraction_step import (
    AUTOENCODERS,
    USERS,
    SERVICES,
    NUM_OF_USERS,
    NUM_OF_SERVICES,
)
from recommender.engines.autoencoders.training.data_preparation_step import (
    precalc_users_and_service_tensors,
    split_autoencoder_datasets,
    DataPreparationStep,
    TRAIN_DS_SIZE,
    VALID_DS_SIZE,
    TRAIN,
    VALID,
    TEST,
)
from recommender.engines.autoencoders.training.model_training_step import (
    ENCODER_LAYER_SIZES,
    DECODER_LAYER_SIZES,
    USER_EMBEDDING_DIM,
    SERVICE_EMBEDDING_DIM,
    LR,
    EPOCHS,
    LOSS,
    TRAINING_TIME,
    MODEL,
    EMBEDDER,
    DATASET,
    get_train_and_valid_datasets,
    create_autoencoder_model,
    perform_training,
    autoencoder_loss_function,
    evaluate_autoencoder,
)
from recommender.engines.autoencoders.training.model_evaluation_step import (
    METRICS,
    BATCH_SIZE,
    ModelEvaluationStep,
)
from recommender.engines.autoencoders.ml_components.embedder import Embedder
from recommender.engines.constants import DEVICE, WRITER, VERBOSE

USER_FEATURES_DIM = "user_features_dim"
SERVICE_FEATURES_DIM = "service_features_dim"


@pytest.fixture
def simulate_data_extraction_step(
    mongo, generate_users_and_services, ae_pipeline_config
):
    """Simulate the autoencoders data extraction step"""

    return process_data_extraction_data()


@pytest.fixture
def simulate_invalid_data_extraction_step(
    mongo, generate_invalid_users_and_services, ae_pipeline_config
):
    """Simulate the invalid autoencoders data extraction step"""

    return process_data_extraction_data()


def get_empty_data_and_details():
    data = {AUTOENCODERS: {USERS: [], SERVICES: []}}
    details = {USERS: {}, SERVICES: {}}

    return data, details


def process_data_extraction_data():
    data, details = get_empty_data_and_details()

    # Get users and services
    users = User.objects.order_by("-id")
    services = Service.objects.order_by("-id")

    num_of_usr = users.count()
    num_of_srv = services.count()

    data[AUTOENCODERS][USERS] = list(users)
    data[AUTOENCODERS][SERVICES] = list(services)

    details[USERS][NUM_OF_USERS] = num_of_usr
    details[SERVICES][NUM_OF_SERVICES] = num_of_srv

    return data, details


@pytest.fixture
def simulate_data_preparation_step(
    mongo, simulate_data_extraction_step, ae_pipeline_config
):
    """Simulate the autoencoders data preparation step"""

    data = {AUTOENCODERS: {}}
    details = {}
    config = ae_pipeline_config[DataPreparationStep.__name__]
    train_ds_size = config[TRAIN_DS_SIZE]
    valid_ds_size = config[VALID_DS_SIZE]
    device = ae_pipeline_config[DEVICE]

    # Simulate data extraction step
    data_ext_step, _ = simulate_data_extraction_step
    raw_data = data_ext_step[AUTOENCODERS]

    tensors = precalc_users_and_service_tensors(raw_data)

    for collection_name, dataset in tensors.items():
        splitted_ds = split_autoencoder_datasets(
            dataset,
            train_ds_size,
            valid_ds_size,
            device,
        )

        data[AUTOENCODERS][collection_name] = splitted_ds

    return data, details


@pytest.fixture
def training_data(mongo, simulate_data_preparation_step, ae_pipeline_config):
    """Get the essential data for a model training"""
    training_data = {}

    # Simulate data preparation step
    data_prep_step, _ = simulate_data_preparation_step
    data_prep_step = data_prep_step[AUTOENCODERS]

    # Encoder/decoder's layers sizes
    training_data[ENCODER_LAYER_SIZES] = (128, 64)
    training_data[DECODER_LAYER_SIZES] = (64, 128)

    # Features dims
    training_data[USER_FEATURES_DIM] = len(data_prep_step[USERS][TRAIN][0][0])
    training_data[SERVICE_FEATURES_DIM] = len(data_prep_step[SERVICES][TRAIN][0][0])

    # Embedding dims
    training_data[USER_EMBEDDING_DIM] = randint(32, 256)
    training_data[SERVICE_EMBEDDING_DIM] = randint(32, 256)

    # Batch sizes
    user_batch_size = randint(32, 256)
    service_batch_size = randint(32, 256)

    # Train and valid datasets
    datasets = get_train_and_valid_datasets(
        data_prep_step, user_batch_size, service_batch_size
    )
    training_data.update(datasets)

    training_data[LR] = uniform(0.001, 0.1)
    training_data[EPOCHS] = randint(2, 10)
    training_data[DEVICE] = ae_pipeline_config[DEVICE]
    training_data[VERBOSE] = ae_pipeline_config[VERBOSE]
    training_data[WRITER] = ae_pipeline_config[WRITER]

    return training_data


@pytest.fixture
def get_autoencoder_models(training_data):
    """Create autoencoders models for users and services"""
    ae_models = {USERS: {}, SERVICES: {}}
    data = training_data

    for col in ae_models:
        features_dim = (
            data[USER_FEATURES_DIM] if col == USERS else data[SERVICE_FEATURES_DIM]
        )
        embedding_dim = (
            data[USER_EMBEDDING_DIM] if col == USERS else data[SERVICE_EMBEDDING_DIM]
        )
        train_ds_dl = data[USERS][TRAIN] if col == USERS else data[SERVICES][TRAIN]

        ae_model = create_autoencoder_model(
            collection_name=col,
            encoder_layer_sizes=data[ENCODER_LAYER_SIZES],
            decoder_layer_sizes=data[DECODER_LAYER_SIZES],
            features_dim=features_dim,
            embedding_dim=embedding_dim,
            writer=data[WRITER],
            train_ds_dl=train_ds_dl,
            device=data[DEVICE],
        )

        ae_models[col] = ae_model

    return ae_models


@pytest.fixture
def simulate_model_training_step(training_data, simulate_data_preparation_step):
    """Simulate the AE model training step"""
    data = training_data

    data_model_train_step = {USERS: {}, SERVICES: {}}
    details = {USERS: {}, SERVICES: {}}

    for collection in data_model_train_step:
        features_dim = (
            data[USER_FEATURES_DIM]
            if collection == USERS
            else data[SERVICE_FEATURES_DIM]
        )
        embedding_dim = (
            data[USER_EMBEDDING_DIM]
            if collection == USERS
            else data[SERVICE_EMBEDDING_DIM]
        )

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
        data_model_train_step[collection][MODEL] = model
        data_model_train_step[collection][EMBEDDER] = Embedder(model)
        data_model_train_step[collection][DATASET] = simulate_data_preparation_step[0][
            AUTOENCODERS
        ][collection]

        details[collection][LOSS] = loss
        details[collection][TRAINING_TIME] = timer

    return data_model_train_step, details


@pytest.fixture
def simulate_model_evaluation_step(simulate_model_training_step, ae_pipeline_config):
    """Simulate the AE model evaluation step"""
    data, _ = simulate_model_training_step

    metrics = {
        USERS: {TRAIN: {}, VALID: {}, TEST: {}},
        SERVICES: {TRAIN: {}, VALID: {}, TEST: {}},
    }

    for collection_name, datasets in data.items():
        model = datasets[MODEL]
        device = ae_pipeline_config[DEVICE]
        batch_size = ae_pipeline_config[ModelEvaluationStep.__name__][BATCH_SIZE]

        for split, dataset in datasets[DATASET].items():
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            loss = evaluate_autoencoder(
                model, dataloader, autoencoder_loss_function, device
            )
            metrics[collection_name][split] = loss

    details = {METRICS: metrics}
    data[METRICS] = metrics

    return data, details
