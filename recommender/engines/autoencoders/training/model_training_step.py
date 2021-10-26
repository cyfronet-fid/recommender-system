# pylint: disable=too-many-arguments, too-many-instance-attributes, too-many-locals
# pylint: disable=not-callable, too-many-branches
"""Autoencoder Model Training Step."""

import time
from copy import deepcopy
from typing import Tuple
from tqdm.auto import tqdm

import torch
from torch.nn.utils import clip_grad_norm_
from torch.nn import CosineEmbeddingLoss
from torch.utils.data import DataLoader
from torch.optim import Adam

from recommender.engines.autoencoders.ml_components.autoencoder import AutoEncoder
from recommender.engines.constants import WRITER, VERBOSE, DEVICE
from recommender.engines.base.base_steps import ModelTrainingStep
from recommender.engines.autoencoders.training.data_preparation_step import TRAIN
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
LOSS = "loss"
DATASET = "dataset"


def create_autoencoder_model(
    collection_name,
    features_dim,
    embedding_dim,
    writer=None,
    train_ds_dl=None,
    device=torch.device("cpu"),
):
    """This function should be used to instantiate and Autoencoder rather
    than direct autoencoder class"""

    if collection_name == USERS:
        model = AutoEncoder(features_dim=features_dim, embedding_dim=embedding_dim).to(
            device
        )
    elif collection_name == SERVICES:
        model = AutoEncoder(features_dim=features_dim, embedding_dim=embedding_dim).to(
            device
        )
    else:
        raise ValueError

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


def evaluate_autoencoder(model, dataloader, loss_function, device):
    """Evaluate autoencoder"""

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            features = batch[0].float().to(device)
            preds = model(features).to(device)
            loss = loss_function(features, preds)
        return loss.item()


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
        with tqdm(train_ds_dl, unit="batch", disable=(not verbose)) as tepoch:
            model.train()
            for batch in tepoch:
                tepoch.set_description(f"[Epoch {epoch}]")

                features = batch[0].float().to(device)
                reconstructions = model(features).to(device)
                loss = loss_function(reconstructions, features)
                loss.backward()

                clip_grad_norm_(model.parameters(), 0.3)
                optimizer.step()
                optimizer.zero_grad()

                loss = loss.item()
                tepoch.set_postfix(loss=loss)

            val_loss = evaluate_autoencoder(model, valid_ds_dl, loss_function, device)

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
                writer.flush()

            tepoch.set_postfix(
                loss=loss, val_loss=val_loss, best_model=str(best_model_flag)
            )

    end = time.time()
    timer = end - start

    if verbose:
        print(f"Total training time: {end - start}")

    return best_model, timer


def perform_training(
    collection_name,
    train_ds_dl,
    embedding_dim,
    features_dim,
    writer,
    device,
    learning_rate,
    epochs,
    verbose,
):
    """Perform training"""

    autoencoder_model = create_autoencoder_model(
        collection_name,
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
        writer=writer,
        save_period=10,
        verbose=verbose,
        device=device,
    )

    loss = evaluate_autoencoder(
        trained_autoencoder_model,
        train_ds_dl,
        autoencoder_loss_function,
        device,
    )

    print(f"User Autoencoder testing loss: {loss}")

    return trained_autoencoder_model, loss, timer


class AEModelTrainingStep(ModelTrainingStep):
    """Autoencoder model training step"""

    def __init__(self, pipeline_config):
        super().__init__(pipeline_config)
        self.device = self.resolve_constant(DEVICE, torch.device("cpu"))
        self.writer = self.resolve_constant(WRITER)
        self.verbose = self.resolve_constant(VERBOSE, True)
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

        # Users
        user_train_ds = raw_data[USERS][TRAIN]

        user_train_ds_dl = DataLoader(
            user_train_ds, batch_size=self.user_batch_size, shuffle=True
        )

        user_features_dim = len(raw_data[USERS][TRAIN][0][0])
        service_features_dim = len(raw_data[SERVICES][TRAIN][0][0])

        user_model, user_loss, user_timer = perform_training(
            collection_name=USERS,
            train_ds_dl=user_train_ds_dl,
            embedding_dim=self.user_embedding_dim,
            features_dim=user_features_dim,
            writer=self.writer,
            device=self.device,
            learning_rate=self.learning_rate,
            epochs=self.epochs,
            verbose=self.verbose,
        )

        # Services
        service_train_ds = raw_data[SERVICES][TRAIN]

        service_train_ds_dl = DataLoader(
            service_train_ds, batch_size=self.service_batch_size, shuffle=True
        )

        service_model, service_loss, service_timer = perform_training(
            collection_name=SERVICES,
            train_ds_dl=service_train_ds_dl,
            embedding_dim=self.service_embedding_dim,
            features_dim=service_features_dim,
            writer=self.writer,
            device=self.device,
            learning_rate=self.learning_rate,
            epochs=self.epochs,
            verbose=self.verbose,
        )

        self.trained_user_embedder = Embedder(user_model)
        self.trained_service_embedder = Embedder(service_model)

        details = {}
        user_autoencoder_details = {LOSS: user_loss, TRAINING_TIME: user_timer}
        service_autoencoder_details = {LOSS: service_loss, TRAINING_TIME: service_timer}

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

    def save(self):
        """Save a model"""
        self.trained_user_embedder.save(version=USER_EMBEDDER)
        self.trained_service_embedder.save(version=SERVICE_EMBEDDER)
