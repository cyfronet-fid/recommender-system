# pylint: disable=redefined-builtin, no-member, redefined-outer-name, missing-function-docstring, too-many-arguments
# pylint: disable=too-many-locals

"""Functions used for training end evaluation of the Naural Collaborative
 filtering model"""

import time
from copy import deepcopy

import torch
from tqdm.auto import tqdm


def accuracy_function(preds, labels):
    """Calculate accuracy for given predictions and labels tensors"""

    rounded_preds = torch.round(torch.reshape(preds, (-1,)))
    reshaped_labels = torch.reshape(labels, (-1,))
    all = len(reshaped_labels)
    matching = torch.sum(rounded_preds == reshaped_labels).item()
    return matching / all


def evaluate_ncf(
    model, dataloader, loss_function, accuracy_function, device=torch.device("cpu")
):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            users_ids = batch["users_ids"].to(device)
            users_contents = batch["users"].to(device)
            services_ids = batch["services_ids"].to(device)
            services_contents = batch["services"].to(device)
            labels = batch["labels"].to(device)

            preds = model(users_ids, users_contents, services_ids, services_contents)
            loss = loss_function(preds, labels)
            acc = accuracy_function(preds, labels)
        return loss.item(), acc


def train_ncf(
    model,
    optimizer,
    loss_function,
    epochs,
    train_ds_dl,
    valid_ds_dl,
    save_period=10,
    writer=None,
    verbose=False,
    device=torch.device("cpu"),
):
    model = model.to(device)

    best_model = deepcopy(model)
    best_model_val_loss = float("+Inf")

    start = time.time()
    for epoch in range(epochs):
        with tqdm(train_ds_dl, unit="batch", disable=(not verbose)) as tepoch:
            model.train()
            for batch in tepoch:
                tepoch.set_description(f"[Epoch {epoch}]")

                users_contents = batch["users"].to(device)
                users_ids = batch["users_ids"].to(device)
                services_contents = batch["services"].to(device)
                services_ids = batch["services_ids"].to(device)

                labels = batch["labels"].to(device)

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

    if verbose:
        print(f"Total training time: {end - start}")

    return best_model
