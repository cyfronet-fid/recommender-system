# pylint: disable=line-too-long, redefined-builtin, redefined-outer-name
# pylint: disable=too-many-arguments, too-many-locals, too-many-instance-attributes

"""Neural Collaborative Filtering Model Training Step."""

import time
from copy import deepcopy
from typing import Tuple, Dict, Callable

import torch
from numpy import ndarray
from torch import Tensor
from torch.nn import BCELoss
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from recommender.engines.base.base_steps import ModelTrainingStep
from recommender.engines.ncf.ml_components.neural_collaborative_filtering import (
    NeuralCollaborativeFilteringModel,
    NEURAL_CF,
)
from recommender.engines.ncf.training.data_extraction_step import (
    USERS_MAX_ID,
    SERVICES_MAX_ID,
    USERS,
    SERVICES,
)
from recommender.engines.ncf.training.data_preparation_step import (
    DATASETS,
    USERS_IDS,
    SERVICES_IDS,
    LABELS,
    TRAIN,
    VALID,
)
from recommender.engines.constants import WRITER, VERBOSE, DEVICE
from recommender.engines.ncf.ml_components.tensor_dict_dataset import (
    TensorDictDataset,
)

BATCH_SIZE = "batch_size"
MF_EMBEDDING_DIM = "mf_embedding_dim"
USER_IDS_EMBEDDING_DIM = "user_ids_embedding_dim"
SERVICE_IDS_EMBEDDING_DIM = "service_ids_embedding_dim"
MLP_LAYERS_SPEC = "mlp_layers_spec"
CONTENT_MLP_LAYERS_SPEC = "content_mlp_layers_spec"
UE = "user_embedding_dim"
SE = "service_embedding_dim"
OPTIMIZER = "optimizer"
OPTIMIZER_PARAMS = "optimizer_params"
EPOCHS = "epochs"
MODEL = "model"
LOSS_FUNCTION = "loss_function"
TRAINING_TIME = "training_time"


def accuracy_function(preds: Tensor, labels: Tensor) -> float:
    """Calculate accuracy for given predictions and labels tensors."""

    rounded_preds = torch.round(torch.reshape(preds, (-1,)))
    reshaped_labels = torch.reshape(labels, (-1,))
    all = len(reshaped_labels)
    matching = torch.sum(rounded_preds == reshaped_labels).item()

    return matching / all


def evaluate_ncf(
    model: NeuralCollaborativeFilteringModel,
    dataloader: DataLoader,
    loss_function: Callable[[Tensor, Tensor], Tensor],
    accuracy_function: Callable[[Tensor, Tensor], float],
    device: torch.device = torch.device("cpu"),
) -> Tuple[float, float]:
    """Evaluate Neural Collaborative Filtering Model"""

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            users_ids = batch[USERS_IDS].to(device)
            users_contents = batch[USERS].to(device)
            services_ids = batch[SERVICES_IDS].to(device)
            services_contents = batch[SERVICES].to(device)
            labels = batch[LABELS].to(device)

            preds = model(users_ids, users_contents, services_ids, services_contents)
            loss = loss_function(preds, labels)
            acc = accuracy_function(preds, labels)
        return loss.item(), acc


def train_ncf(
    model: NeuralCollaborativeFilteringModel,
    optimizer: Optimizer,
    loss_function: Callable[[Tensor, Tensor], Tensor],
    epochs: int,
    train_ds_dl: DataLoader,
    valid_ds_dl: DataLoader,
    save_period: int = 10,
    writer: SummaryWriter = None,
    verbose: bool = False,
    device: torch.device = torch.device("cpu"),
) -> Tuple[NeuralCollaborativeFilteringModel, float]:
    """Main training function"""

    model = model.to(device)

    best_model = deepcopy(model)
    best_model_val_loss = float("+Inf")

    start = time.time()
    for epoch in range(epochs):
        with tqdm(train_ds_dl, unit="batch", disable=(not verbose)) as tepoch:
            model.train()
            for batch in tepoch:
                tepoch.set_description(f"[Epoch {epoch}]")

                users_contents = batch[USERS].to(device)
                users_ids = batch[USERS_IDS].to(device)
                services_contents = batch[SERVICES].to(device)
                services_ids = batch[SERVICES_IDS].to(device)

                labels = batch[LABELS].to(device)

                preds = model(
                    users_ids, users_contents, services_ids, services_contents
                )

                loss = loss_function(preds, labels)
                acc = accuracy_function(preds, labels)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()
                optimizer.zero_grad()

                loss = loss.item()
                tepoch.set_postfix(loss=loss, acc=acc)

            val_loss, val_acc = evaluate_ncf(
                model, valid_ds_dl, loss_function, accuracy_function, device
            )

            best_model_flag = False
            if epoch % save_period == 0:
                if val_loss < best_model_val_loss:
                    best_model_val_loss = val_loss
                    best_model = deepcopy(model)
                    best_model_flag = True

            if writer is not None:
                writer.add_scalars("Loss", {"train": loss, "valid": val_loss}, epoch)
                writer.add_scalars("Accuracy", {"train": acc, "valid": val_acc}, epoch)
                writer.flush()

            tepoch.set_postfix(
                loss=loss,
                acc=acc,
                val_loss=val_loss,
                val_acc=val_acc,
                best_model=str(best_model_flag),
            )

    end = time.time()
    execution_time = end - start

    return best_model, execution_time


def get_preds_for_ds(
    model: NeuralCollaborativeFilteringModel,
    dataset: TensorDictDataset,
    device: torch.device = torch.device("cpu"),
) -> Tuple[ndarray, ndarray]:
    """Used for getting predictions on pytorch dataset."""

    model = model.to(device)

    all_samples = dataset[:]

    users_ids = all_samples[USERS_IDS].to(device)
    users_contents = all_samples[USERS].to(device)
    services_ids = all_samples[SERVICES_IDS].to(device)
    services_contents = all_samples[SERVICES].to(device)
    labels = all_samples[LABELS].to(device)
    preds = model(users_ids, users_contents, services_ids, services_contents)

    labels = labels.detach().numpy()
    preds = preds.detach().numpy()

    return labels, preds


class NCFModelTrainingStep(ModelTrainingStep):
    """Neural Collaborative Filtering Model Training Step."""

    def __init__(self, pipeline_config):
        super().__init__(pipeline_config)
        self.device = self.resolve_constant(DEVICE, torch.device("cpu"))
        self.writer = self.resolve_constant(WRITER)
        self.verbose = self.resolve_constant(VERBOSE, False)

        self.batch_size = self.resolve_constant(BATCH_SIZE, 64)

        self.mf_embedding_dim = self.resolve_constant(MF_EMBEDDING_DIM, 64)
        self.users_ids_embedding_dim = self.resolve_constant(USER_IDS_EMBEDDING_DIM, 64)
        self.user_emb_dim = self.resolve_constant(UE)
        self.service_emb_dim = self.resolve_constant(SE)
        self.services_ids_embedding_dim = self.resolve_constant(
            SERVICE_IDS_EMBEDDING_DIM, 64
        )
        self.mlp_layers_spec = self.resolve_constant(MLP_LAYERS_SPEC, (64, 32, 16, 8))
        self.content_mlp_layers_spec = self.resolve_constant(
            CONTENT_MLP_LAYERS_SPEC, (128, 64, 32)
        )
        self.optimizer = self.resolve_constant(OPTIMIZER, Adam)
        self.optimizer_params = self.resolve_constant(OPTIMIZER_PARAMS, {"lr": 0.01})
        self.epochs = self.resolve_constant(EPOCHS, 500)
        self.loss_function = self.resolve_constant(LOSS_FUNCTION, BCELoss())

        self.trained_model = None

    def __call__(self, data: Dict = None) -> Tuple[Dict, Dict]:
        """Perform model training."""

        train_ds = data[DATASETS][TRAIN]
        valid_ds = data[DATASETS][VALID]

        train_ds_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        valid_ds_dl = DataLoader(valid_ds, batch_size=self.batch_size, shuffle=True)

        model = NeuralCollaborativeFilteringModel(
            users_max_id=data[USERS_MAX_ID],
            services_max_id=data[SERVICES_MAX_ID],
            mf_embedding_dim=self.mf_embedding_dim,
            user_ids_embedding_dim=self.users_ids_embedding_dim,
            service_ids_embedding_dim=self.services_ids_embedding_dim,
            user_emb_dim=self.user_emb_dim,
            service_emb_dim=self.service_emb_dim,
            mlp_layers_spec=self.mlp_layers_spec,
            content_mlp_layers_spec=self.content_mlp_layers_spec,
        ).to(self.device)

        if self.writer is not None:
            batch = next(iter(train_ds_dl))
            example_input = (
                batch[USERS_IDS].to(self.device),
                batch[USERS].to(self.device),
                batch[SERVICES_IDS].to(self.device),
                batch[SERVICES].to(self.device),
            )

            self.writer.add_graph(model, example_input)

        ncf_optimizer = self.optimizer(model.parameters(), **self.optimizer_params)

        self.trained_model, training_time = train_ncf(
            model=model,
            optimizer=ncf_optimizer,
            loss_function=self.loss_function,
            epochs=self.epochs,
            train_ds_dl=train_ds_dl,
            valid_ds_dl=valid_ds_dl,
            save_period=10,
            writer=self.writer,
            verbose=self.verbose,
            device=self.device,
        )

        details = {TRAINING_TIME: training_time}

        data = {DATASETS: data[DATASETS], MODEL: self.trained_model}

        return data, details

    def save(self):
        """Save trained model with the proper version."""

        self.trained_model.save(version=NEURAL_CF)
