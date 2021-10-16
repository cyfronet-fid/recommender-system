# pylint: disable=too-many-instance-attributes, invalid-name, no-member
# pylint: disable=too-many-arguments, missing-function-docstring, line-too-long

"""This module contain neural colaborative filtering model
 - the essential part of the pre agent"""

import torch
from torch.nn import Module, Linear

from recommender.engine.agents.pre_agent.models.content_mlp import ContentMLP
from recommender.engine.agents.pre_agent.models.gmf import GMF
from recommender.engine.agents.pre_agent.models.mlp import MLP
from recommender.engines.persistent_mixin import Persistent

NEURAL_CF = "Neural Collaborative Filtering Model"


class NeuralColaborativeFilteringModel(Module, Persistent):
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


def create_nfc_model(
    users_max_id,
    services_max_id,
    mf_embedding_dim,
    user_ids_embedding_dim,
    service_ids_embedding_dim,
    user_emb_dim,
    service_emb_dim,
    mlp_layers_spec,
    content_mlp_layers_spec,
    writer=None,
    train_ds_dl=None,
    device=torch.device("cpu"),
):
    """It should be used for instantiating Neural Collaborative Model rather than direct class"""

    model = NeuralColaborativeFilteringModel(
        users_max_id=users_max_id,
        services_max_id=services_max_id,
        mf_embedding_dim=mf_embedding_dim,
        user_ids_embedding_dim=user_ids_embedding_dim,
        service_ids_embedding_dim=service_ids_embedding_dim,
        user_emb_dim=user_emb_dim,
        service_emb_dim=service_emb_dim,
        mlp_layers_spec=mlp_layers_spec,
        content_mlp_layers_spec=content_mlp_layers_spec,
    ).to(device)

    if writer is not None and train_ds_dl is not None:
        batch = next(iter(train_ds_dl))
        example_input = (
            batch["users_ids"].to(device),
            batch["users"].to(device),
            batch["services_ids"].to(device),
            batch["services"].to(device),
        )

        writer.add_graph(model, example_input)

    return model


def get_preds_for_ds(model, dataset, device=torch.device("cpu")):
    """Used for getting predictions on pytorch dataset"""

    model = model.to(device)

    all_samples = dataset[:]

    users_ids = all_samples["users_ids"].to(device)
    users_contents = all_samples["users"].to(device)
    services_ids = all_samples["services_ids"].to(device)
    services_contents = all_samples["services"].to(device)
    labels = all_samples["labels"].to(device)
    preds = model(users_ids, users_contents, services_ids, services_contents)

    labels = labels.detach().numpy()
    preds = preds.detach().numpy()

    return labels, preds
