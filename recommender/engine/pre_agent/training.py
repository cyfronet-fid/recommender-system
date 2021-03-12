# pylint: disable=redefined-builtin, no-member, redefined-outer-name,
# pylint: disable=too-many-arguments, too-many-locals, invalid-name

"""Training and evaluation related functions"""

import time
import torch
from torch.nn import BCELoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from matplotlib import pyplot as plt

from recommender import NEURAL_CF
from recommender.engine.pre_agent.datasets import (
    load_last_dataset,
    TRAIN_DS_NAME,
    VALID_DS_NAME,
    TEST_DS_NAME,
)
from recommender.engine.pre_agent.models import (
    NeuralColaborativeFilteringModel,
    save_module,
)
from recommender.engine.pre_agent.preprocessing import LABELS, USERS, SERVICES


def accuracy_function(preds, labels):
    """Calculate accuracy for given predictions and labels tensors"""

    rounded_preds = torch.round(torch.reshape(preds, (-1,)))
    reshaped_labels = torch.reshape(labels, (-1,))
    all = len(reshaped_labels)
    matching = torch.sum(rounded_preds == reshaped_labels).item()
    return matching / all


def evaluate(model, dataloader, loss_function, accuracy_function):
    """This function is implicitly used for evaluating model during trainng on
    validation set and it can be also explicitly used for evaluation model on
    test set"""
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            labels = batch[LABELS]
            preds = model(batch[USERS], batch[SERVICES])
            loss = loss_function(preds, labels).item()
            acc = accuracy_function(preds, labels)
            return loss, acc


def train(
    model, epochs, train_ds_dl, valid_ds_dl, loss_function, accuracy_function, optimizer
):
    """It can be used for model training. It provides validation, progress
    bars and train/val accuracy and loss tracking. It loosly match keras
    .fit method."""

    losses = []
    accuracies = []

    val_losses = []
    val_accuracies = []

    start = time.time()
    for epoch in range(epochs):
        with tqdm(train_ds_dl, unit="batch") as tepoch:
            model.train()
            for batch in tepoch:
                tepoch.set_description(f"[Epoch {epoch}]")
                labels = batch[LABELS]
                preds = model(batch[USERS], batch[SERVICES])

                loss = loss_function(preds, labels)
                acc = accuracy_function(preds, labels)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                loss = loss.item()
                tepoch.set_postfix(loss=loss, acc=acc)

            losses.append(loss)
            accuracies.append(acc)

            val_loss, val_acc = evaluate(
                model, valid_ds_dl, loss_function, accuracy_function
            )
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            tepoch.set_postfix(loss=loss, acc=acc, val_loss=val_loss, val_acc=val_acc)

    end = time.time()
    print(f"Total training time: {end - start}")

    return losses, accuracies, val_losses, val_accuracies


def pre_agent_training():
    """Main full training function. It load datasets, creates model, loss
    function and optimizer. It trains model, show train/valid accuracy/loss
    and evaluate model on the test set. In the end it saves model.

    This function is used in all tasks and is also called in the training endpoint.
    """

    # Get datasets and dataloaders
    train_ds = load_last_dataset(TRAIN_DS_NAME)
    valid_ds = load_last_dataset(VALID_DS_NAME)
    test_ds = load_last_dataset(TEST_DS_NAME)

    BATCH_SIZE = 32

    train_ds_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    valid_ds_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_ds_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)

    # Model
    USER_FEATURES_DIM = train_ds[0][USERS].shape[0]
    SERVICE_FEATURES_DIM = train_ds[0][SERVICES].shape[0]
    USER_EMBEDDING_DIM = 32
    SERVICE_EMBEDDING_DIM = 64

    neural_cf_model = NeuralColaborativeFilteringModel(
        user_features_dim=USER_FEATURES_DIM,
        user_embedding_dim=USER_EMBEDDING_DIM,
        service_features_dim=SERVICE_FEATURES_DIM,
        service_embedding_dim=SERVICE_EMBEDDING_DIM,
    )

    loss_function = BCELoss()

    LR = 0.01
    optimizer = SGD(neural_cf_model.parameters(), lr=LR)

    EPOCHS = 2000

    losses, accuracies, val_losses, val_accuracies = train(
        model=neural_cf_model,
        epochs=EPOCHS,
        train_ds_dl=train_ds_dl,
        valid_ds_dl=valid_ds_dl,
        loss_function=loss_function,
        accuracy_function=accuracy_function,
        optimizer=optimizer,
    )

    plt.plot([float(loss) for loss in losses])
    plt.plot([float(loss) for loss in val_losses], color="red")
    plt.show()

    plt.plot([float(acc) for acc in accuracies])
    plt.plot([float(acc) for acc in val_accuracies], color="red")
    plt.show()

    loss, acc = evaluate(neural_cf_model, test_ds_dl, loss_function, accuracy_function)
    print(f"Testing loss: {loss}, testing accuracy: {acc}")

    save_module(neural_cf_model, name=NEURAL_CF)
