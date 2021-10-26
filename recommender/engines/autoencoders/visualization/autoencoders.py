# pylint: disable=no-member, inconsistent-return-statements

"""Functions for autoencoders reconstructions visualizations"""

import random

import numpy as np
import torch

import matplotlib.pyplot as plt


def get_random_reconstructions(model, dataset, k=5, device=torch.device("cpu")):
    """Given autoencoder model and appropriate dataset it returns features
    and reconstructions for k random examples"""

    k = min(len(dataset), k)

    subset = random.sample(list(dataset[:][0]), k=k)
    raw_features = [tensor.to(device) for tensor in subset]
    features = torch.stack(raw_features).to(device)
    reconstructions = model(features)

    return features, reconstructions


def tensor2img(features):
    """Convert autoencoder features tensor to the image format"""

    size = int(np.ceil(np.sqrt(len(features))))
    im_tensor = torch.zeros(size, size)
    for i in range(size):
        for j in range(size):
            idx = i * size + j
            if idx >= len(features):
                return im_tensor.detach().cpu().numpy()
            im_tensor[i, j] = features[idx].item()


def plot_reconstructions(features, reconstructions):
    """Plot features and their reconstructions"""

    cols_number = len(features)
    fig, axes = plt.subplots(2, cols_number, figsize=(4 * cols_number, 12))
    for i, (feature, reconstruction) in enumerate(zip(features, reconstructions)):
        if cols_number > 1:
            axes[0, i].imshow(tensor2img(feature))
            axes[1, i].imshow(tensor2img(reconstruction))
        else:
            axes[0].imshow(tensor2img(feature))
            axes[1].imshow(tensor2img(reconstruction))

    if cols_number > 1:
        axes[0, 0].set_title("original\ntensor\nrepresentation")
        axes[1, 0].set_title("reconstructed\ntensor\nrepresentation")
    else:
        axes[0].set_title("original\ntensor\nrepresentation")
        axes[1].set_title("reconstructed\ntensor\nrepresentation")
    fig.tight_layout()
    plt.show()
