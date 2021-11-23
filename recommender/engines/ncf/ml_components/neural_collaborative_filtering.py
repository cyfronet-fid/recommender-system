# pylint: disable=too-many-instance-attributes, invalid-name, no-member
# pylint: disable=too-many-arguments, missing-function-docstring, line-too-long

"""This module contain neural colaborative filtering model
 - the essential part of the pre agent"""

import torch
from torch.nn import Module, Linear

from recommender.engines.ncf.ml_components.content_mlp import ContentMLP
from recommender.engines.ncf.ml_components.gmf import GMF
from recommender.engines.ncf.ml_components.mlp import MLP
from recommender.engines.persistent_mixin import Persistent

NEURAL_CF = "Neural Collaborative Filtering Model"


class NeuralCollaborativeFilteringModel(Module, Persistent):
    """Pytorch module containing neural network of the neural colaborative filtering model"""

    def __init__(
        self,
        users_max_id,
        services_max_id,
        mf_embedding_dim,
        user_ids_embedding_dim,
        service_ids_embedding_dim,
        user_emb_dim,
        service_emb_dim,
        mlp_layers_spec,
        content_mlp_layers_spec,
    ):
        super().__init__()
        self.gmf = GMF(users_max_id, services_max_id, mf_embedding_dim)
        self.mlp = MLP(
            users_max_id,
            services_max_id,
            user_ids_embedding_dim,
            service_ids_embedding_dim,
            mlp_layers_spec,
        )
        self.content_mlp = ContentMLP(
            content_mlp_layers_spec, user_emb_dim, service_emb_dim
        )
        self.fc = Linear(
            mf_embedding_dim + mlp_layers_spec[-1] + content_mlp_layers_spec[-1], 1
        )

    def forward(self, users_ids, users_contents, services_ids, services_contents):
        """Method used for performing forward propagation"""

        gmf_output = self.gmf(users_ids, services_ids)
        mlp_output = self.mlp(users_ids, services_ids)
        content_mlp_output = self.content_mlp(users_contents, services_contents)

        x = torch.cat([gmf_output, mlp_output, content_mlp_output], dim=1)
        output = torch.sigmoid(self.fc(x))

        return output
