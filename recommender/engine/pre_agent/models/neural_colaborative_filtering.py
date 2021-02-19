# pylint: disable=too-many-instance-attributes, invalid-name, no-member


"""This module contain neural colaborative filtering model
 - the essential part of the pre agent"""


import torch
from torch.nn import Module, Linear
import torch.nn.functional as F

from recommender.engine.pre_agent.models.user_embedder import UserEmbedder
from recommender.engine.pre_agent.models.service_embedder import ServiceEmbedder

NEURAL_CF = "neural_cf"


class NeuralColaborativeFilteringModel(Module):
    """Pytorch module containing neural network of the
    neural colaborative filtering model"""

    def __init__(
        self,
        user_features_dim,
        user_embedding_dim,
        service_features_dim,
        service_embedding_dim,
    ):
        super().__init__()
        self.user_features_dim = user_features_dim
        self.user_embedding_dim = user_embedding_dim
        self.service_features_dim = service_features_dim
        self.service_embedding_dim = service_embedding_dim

        self.user_embedder = UserEmbedder(user_features_dim, user_embedding_dim)
        self.service_embedder = ServiceEmbedder(
            service_features_dim, service_embedding_dim
        )

        embedding_dim = user_embedding_dim + service_embedding_dim

        self.fc1 = Linear(embedding_dim, 8)
        self.fc2 = Linear(8, 8)
        self.fc3 = Linear(8, 1)

    def forward(self, user_features, service_features):
        """Method used for performing forward propagation"""

        embedded_user = self.user_embedder(user_features)
        embedded_service = self.service_embedder(service_features)

        x = torch.cat((embedded_user, embedded_service), dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))

        return x
