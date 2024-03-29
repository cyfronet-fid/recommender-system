# pylint: disable=too-many-arguments, too-many-instance-attributes, too-many-locals
# pylint: disable=not-callable, too-many-branches
"""Autoencoder Model Training Step."""

import time
from copy import deepcopy
from typing import Tuple, Callable
from tqdm.auto import tqdm

import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.nn import CosineEmbeddingLoss
from torch.utils.data import DataLoader
from torch.optim import Adam

from recommender.engines.autoencoders.ml_components.autoencoder import AutoEncoder
from recommender.engines.constants import WRITER, VERBOSE, DEVICE, ACCURACY, LOSS
from recommender.engines.base.base_steps import ModelTrainingStep
from recommender.engines.autoencoders.training.data_preparation_step import TRAIN, VALID
from recommender.engines.autoencoders.training.data_extraction_step import (
    AUTOENCODERS,
    USERS,
    SERVICES,
)
from recommender.engines.autoencoders.ml_components.embedder import (
    Embedder,
    SERVICE_EMBEDDER,
    USER_EMBEDDER,
)
from recommender.engines.metadata_creators import accuracy_function
from logger_config import get_logger

ENCODER_LAYER_SIZES = "encoder_layer_sizes"
DECODER_LAYER_SIZES = "decoder_layer_sizes"
USER_BATCH_SIZE = "user_batch_size"
SERVICE_BATCH_SIZE = "service_batch_size"
USER_EMBEDDING_DIM = "user_embedding_dim"
SERVICE_EMBEDDING_DIM = "service_embedding_dim"
LOSS_FUNCTION = "loss_function"
OPTIMIZER = "optimizer"
LR = "learning_rate"
EPOCHS = "epochs"
EMBEDDER = "embedder"
MODEL = "model"
TRAINING_TIME = "training_time"
DATASET = "dataset"

logger = get_logger(__name__)


def get_train_and_valid_datasets(datasets, user_batch_size, service_batch_size):
    """
    Use pytorch's DataLoader on train/valid users/services datasets to enable training
    """

    training_datasets = {USERS: {}, SERVICES: {}}
    for collection in (USERS, SERVICES):
        if collection == USERS:
            batch_size = user_batch_size
        else:
            batch_size = service_batch_size
        for split in (TRAIN, VALID):
            dataset = datasets[collection][split]
            training_datasets[collection][split] = DataLoader(
                dataset, batch_size=batch_size, shuffle=True
            )
    return training_datasets


def create_autoencoder_model(
    collection_name,
    encoder_layer_sizes,
    decoder_layer_sizes,
    features_dim,
    embedding_dim,
    writer=None,
    train_ds_dl=None,
    device=torch.device("cpu"),
):
    """This function should be used to instantiate and Autoencoder rather
    than direct autoencoder class"""

    if collection_name in (USERS, SERVICES):
        model = AutoEncoder(
            features_dim, embedding_dim, encoder_layer_sizes, decoder_layer_sizes
        ).to(device)
    else:
        raise ValueError(f"Collection name not in ({USERS, SERVICES})")

    if writer is not None and train_ds_dl is not None:
        batch = next(iter(train_ds_dl))
        example_input = batch[0].to(device)

        writer.add_graph(model, example_input)

    return model


def autoencoder_loss_function(reconstructions, features):
    """Calculate loss for an autoencoder"""
    cos_emb_loss = CosineEmbeddingLoss(reduction="mean").to(reconstructions.device)
    batch_size = features.shape[0]
    ones = torch.ones(batch_size).to(reconstructions.device)
    return cos_emb_loss(reconstructions, features, ones)


def evaluate_autoencoder(
    model: AutoEncoder,
    dataloader: DataLoader,
    loss_function: Callable[[Tensor, Tensor], Tensor],
    acc_function: Callable[[Tensor, Tensor, bool], float],
    device: torch.device = torch.device("cpu"),
) -> Tuple[float, float]:
    """Evaluate autoencoder"""
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            features = batch[0].float().to(device)
            reconstructions = model(features).to(device)

            loss = loss_function(features, reconstructions)
            acc = acc_function(features, reconstructions, True)
        return round(loss.item(), 3), round(acc, 3)


def train_autoencoder(
    model,
    optimizer,
    loss_function,
    epochs,
    train_ds_dl,
    valid_ds_dl=None,
    save_period=10,
    writer=None,
    verbose=False,
    device=torch.device("cpu"),
):
    """Train autoencoder"""

    if valid_ds_dl is None:
        valid_ds_dl = deepcopy(train_ds_dl)
    model = model.to(device)

    best_model = deepcopy(model)
    best_model_val_loss = float("+Inf")

    start = time.time()
    for epoch in range(epochs):
        epoch_start = time.time()
        with tqdm(train_ds_dl, unit="batch", disable=(not verbose)) as tepoch:
            model.train()
            for batch in tepoch:
                tepoch.set_description(f"[Epoch {epoch}]")

                features = batch[0].float().to(device)
                reconstructions = model(features).to(device)

                loss = loss_function(reconstructions, features)
                acc = accuracy_function(reconstructions, features, labels_rounding=True)
                loss.backward()

                clip_grad_norm_(model.parameters(), 0.3)
                optimizer.step()
                optimizer.zero_grad()

                loss = loss.item()
                tepoch.set_postfix(loss=loss, acc=acc, time=time.time() - epoch_start)

            val_loss, val_acc = evaluate_autoencoder(
                model, valid_ds_dl, loss_function, accuracy_function, device
            )

            best_model_flag = False
            if epoch % save_period == 0:
                if val_loss < best_model_val_loss:
                    best_model_val_loss = val_loss
                    best_model = deepcopy(model)
                    best_model_flag = True

            if writer is not None:
                writer.add_scalars(
                    f"Loss/{model.__class__.__name__}",
                    {"train": loss, "valid": val_loss},
                    epoch,
                )
                writer.add_scalars(
                    f"Accuracy/{model.__class__.__name__}",
                    {"train": acc, "valid": val_acc},
                    epoch,
                )
                writer.flush()

            tepoch.set_postfix(
                loss=loss,
                acc=acc,
                val_loss=val_loss,
                val_acc=val_acc,
                best_model=str(best_model_flag),
            )

    end = time.time()
    timer = round(end - start, 3)

    return best_model, timer


def perform_training(
    encoder_layer_sizes,
    decoder_layer_sizes,
    writer,
    device,
    learning_rate,
    epochs,
    verbose,
    collection_name,
    train_ds_dl,
    valid_ds_dl,
    embedding_dim,
    features_dim,
):
    """Perform training"""
    autoencoder_model = create_autoencoder_model(
        collection_name,
        encoder_layer_sizes=encoder_layer_sizes,
        decoder_layer_sizes=decoder_layer_sizes,
        features_dim=features_dim,
        embedding_dim=embedding_dim,
        writer=writer,
        train_ds_dl=train_ds_dl,
        device=device,
    )

    optimizer = Adam(autoencoder_model.parameters(), lr=learning_rate)

    trained_autoencoder_model, timer = train_autoencoder(
        model=autoencoder_model,
        optimizer=optimizer,
        loss_function=autoencoder_loss_function,
        epochs=epochs,
        train_ds_dl=train_ds_dl,
        valid_ds_dl=valid_ds_dl,
        writer=writer,
        save_period=10,
        verbose=verbose,
        device=device,
    )

    loss, acc = evaluate_autoencoder(
        trained_autoencoder_model,
        train_ds_dl,
        autoencoder_loss_function,
        accuracy_function,
        device,
    )

    if verbose:
        logger.info(
            "Total training time of %s autoencoder: %ss", collection_name, timer
        )
        logger.info(
            "%s autoencoder stats: {loss: %s, accuracy: %s}",
            collection_name.capitalize(),
            loss,
            acc,
        )

    return trained_autoencoder_model, loss, acc, timer


class AEModelTrainingStep(ModelTrainingStep):
    """Autoencoder model training step"""

    def __init__(self, pipeline_config):
        super().__init__(pipeline_config)
        self.device = self.resolve_constant(DEVICE, torch.device("cpu"))
        self.writer = self.resolve_constant(WRITER)
        self.verbose = self.resolve_constant(VERBOSE, True)
        self.encoder_layer_sizes = self.resolve_constant(ENCODER_LAYER_SIZES, (128, 64))
        self.decoder_layer_sizes = self.resolve_constant(DECODER_LAYER_SIZES, (64, 128))
        self.user_batch_size = self.resolve_constant(USER_BATCH_SIZE, 128)
        self.service_batch_size = self.resolve_constant(SERVICE_BATCH_SIZE, 128)
        self.user_embedding_dim = self.resolve_constant(USER_EMBEDDING_DIM, 32)
        self.service_embedding_dim = self.resolve_constant(SERVICE_EMBEDDING_DIM, 64)
        self.learning_rate = self.resolve_constant(LR, 0.01)
        self.epochs = self.resolve_constant(EPOCHS, 500)

        self.trained_user_embedder = None
        self.trained_service_embedder = None

    def __call__(self, data=None) -> Tuple[object, dict]:
        """Perform user and service training"""

        raw_data = data[AUTOENCODERS]
        training_datasets = get_train_and_valid_datasets(
            raw_data, self.user_batch_size, self.service_batch_size
        )

        user_features_dim = len(raw_data[USERS][TRAIN][0][0])
        service_features_dim = len(raw_data[SERVICES][TRAIN][0][0])

        # Users
        user_model, user_loss, user_acc, user_timer = perform_training(
            *self.basic_training_conf(),
            collection_name=USERS,
            train_ds_dl=training_datasets[USERS][TRAIN],
            valid_ds_dl=training_datasets[USERS][VALID],
            embedding_dim=self.user_embedding_dim,
            features_dim=user_features_dim,
        )

        # Services
        service_model, service_loss, service_acc, service_timer = perform_training(
            *self.basic_training_conf(),
            collection_name=SERVICES,
            train_ds_dl=training_datasets[SERVICES][TRAIN],
            valid_ds_dl=training_datasets[SERVICES][VALID],
            embedding_dim=self.service_embedding_dim,
            features_dim=service_features_dim,
        )

        self.trained_user_embedder = Embedder(user_model)
        self.trained_service_embedder = Embedder(service_model)

        details = {}
        user_autoencoder_details = {
            LOSS: user_loss,
            ACCURACY: user_acc,
            TRAINING_TIME: user_timer,
        }
        service_autoencoder_details = {
            LOSS: service_loss,
            ACCURACY: service_acc,
            TRAINING_TIME: service_timer,
        }

        details[USERS] = user_autoencoder_details
        details[SERVICES] = service_autoencoder_details

        data = {}
        user_autoencoder = {
            DATASET: raw_data[USERS],
            EMBEDDER: self.trained_user_embedder,
            MODEL: user_model,
        }

        service_autoencoder = {
            DATASET: raw_data[SERVICES],
            EMBEDDER: self.trained_service_embedder,
            MODEL: service_model,
        }

        data[USERS] = user_autoencoder
        data[SERVICES] = service_autoencoder

        return data, details

    def basic_training_conf(self) -> Tuple:
        """Return a basic training configuration"""
        params = (
            self.encoder_layer_sizes,
            self.decoder_layer_sizes,
            self.writer,
            self.device,
            self.learning_rate,
            self.epochs,
            self.verbose,
        )
        return params

    def save(self) -> None:
        """Save a model"""
        self.trained_user_embedder.save(version=USER_EMBEDDER)
        self.trained_service_embedder.save(version=SERVICE_EMBEDDER)
