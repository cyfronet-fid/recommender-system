"""This module contains functions for creating metadata"""
import torch
from torch import Tensor


def accuracy_function(
    preds: Tensor, labels: Tensor, labels_rounding: bool = False
) -> float:
    """Calculate accuracy for given predictions and labels tensors."""
    rounded_preds = torch.round(torch.reshape(preds, (-1,)))
    reshaped_labels = (
        torch.round(torch.reshape(labels, (-1,)))
        if labels_rounding
        else torch.reshape(labels, (-1,))
    )
    all_elements = len(reshaped_labels)
    matching = torch.sum(rounded_preds == reshaped_labels).item()

    return matching / all_elements
