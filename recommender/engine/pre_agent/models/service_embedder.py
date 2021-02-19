# pylint: disable=invalid-name

"""This module contains Service Embedder"""

from torch.nn import Module, Linear
import torch.nn.functional as F


class ServiceEmbedder(Module):
    """Service embedder is used for transforming high dimensional,
    mostly sparse one-hot representation of the service into
    lower dimensional dense tensor"""

    def __init__(self, features_dim, embedding_dim):
        super().__init__()
        self.fc1 = Linear(features_dim, 8)
        self.fc2 = Linear(8, 8)
        self.fc3 = Linear(8, embedding_dim)

    def forward(self, x):
        """Method used for performing forward propagation"""

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return x
