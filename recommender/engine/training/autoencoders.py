# pylint: disable=missing-function-docstring, no-member, too-many-arguments, too-many-locals
# pylint: disable=not-callable

"""Functions for autoencoders training and evaluation"""

import time
from copy import deepcopy

import torch
from torch.nn import CosineEmbeddingLoss
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm


def evaluate_autoencoder(model, dataloader, loss_function, device):
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            features = batch[0].to(device)
            preds = model(features).to(device)
            loss = loss_function(features, preds)
        return loss.item()


def autoencoder_loss_function(reconstructions, features):
    cos_emb_loss = CosineEmbeddingLoss(reduction="mean").to(reconstructions.device)
    batch_size = features.shape[0]
    ones = torch.ones(batch_size).to(reconstructions.device)
    return cos_emb_loss(reconstructions, features, ones)


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

                features = batch[0].to(device)
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

    if verbose:
        print(f"Total training time: {end - start}")

    return best_model
