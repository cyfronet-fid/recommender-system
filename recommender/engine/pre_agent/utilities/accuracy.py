# pylint: disable=redefined-builtin, no-member

"""Accuracy function for ML training"""

import torch


def accuracy_function(preds, labels):
    """Calculate accuracy for given predictions and labels tensors"""

    rounded_preds = torch.round(torch.reshape(preds, (-1,)))
    reshaped_labels = torch.reshape(labels, (-1,))
    all = len(reshaped_labels)
    matching = torch.sum(rounded_preds == reshaped_labels).item()
    return matching / all
