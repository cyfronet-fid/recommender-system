# pylint: disable-all

import torch

from recommender.engine.agents.panel_id_to_services_number_mapping import PANEL_ID_TO_K
from recommender.engine.agents.rl_agent.models.action_embedder import ActionEmbedder
from recommender.engine.agents.rl_agent.preprocessing.search_data_encoder import (
    SearchDataEncoder,
)
from recommender.engine.preprocessing import load_last_transformer
from recommender.engine.utils import save_module
from recommender.engine.models.autoencoders import (
    ServiceAutoEncoder,
    create_embedder,
    UserAutoEncoder,
    USERS_AUTOENCODER,
    SERVICES_AUTOENCODER,
)
from recommender.engine.agents.rl_agent.preprocessing.searchphrase_encoder import (
    SearchPhraseEncoder,
)
from recommender.engine.agents.rl_agent.preprocessing.state_encoder import StateEncoder
from recommender.engine.preprocessing import (
    precalculate_tensors,
    create_transformer,
    SERVICES,
    USERS,
)
from recommender.engine.agents.rl_agent.preprocessing.filters_encoder import (
    FiltersEncoder,
)
from recommender.models import Service, User
from tests.factories.marketplace import ServiceFactory
from tests.factories.state import StateFactory


def test_state_embedder(mongo):
    # TODO: test the actual state embedder, this is a duplication of state_enoder
    pass
    # UE = 32
    # SE = 128
    # N = 10
    #
    # state = StateFactory(services_history=ServiceFactory.create_batch(N))
    # precalculate_tensors(Service.objects, create_transformer(SERVICES))
    # precalculate_tensors(User.objects, create_transformer(USERS))
    # state.reload()
    #
    # UOH = len(User.objects.first().tensor)
    # SOH = len(Service.objects.first().tensor)
    #
    # # UserEmbedder
    # user_autoencoder = UserAutoEncoder(features_dim=UOH, embedding_dim=UE)
    # user_embedder = create_embedder(user_autoencoder)
    # save_module(module=user_autoencoder, name=USERS_AUTOENCODER)
    #
    # # ServiceEmbedder
    # service_auto_encoder = ServiceAutoEncoder(features_dim=SOH, embedding_dim=SE)
    # service_embedder = create_embedder(service_auto_encoder)
    # save_module(module=service_auto_encoder, name=SERVICES_AUTOENCODER)
    #
    # # state Encoder
    # state_encoder = StateEncoder(
    #     user_embedder=user_embedder,
    #     service_embedder=service_embedder,
    #     search_data_encoder=SearchDataEncoder()
    # )
    #
    # encoded_state = state_encoder(state)
    #
    # assert type(encoded_state) == tuple
    # assert len(encoded_state) == 4
    # for embedding in encoded_state:
    #     assert type(embedding) == torch.Tensor
    #
    # assert encoded_state[0].shape == torch.Size([UE])
    # assert encoded_state[1].shape == torch.Size([N, SE])
    # assert encoded_state[2].shape == torch.Size([SE])
    # assert encoded_state[3].shape[1] == SPE


def test_action_embedder(mongo):
    SE = 128
    for K in list(PANEL_ID_TO_K.values()):
        action_tensor = torch.rand((K, SE))
        action_embedder = ActionEmbedder(SE=SE)
        action_embedder_batch = action_tensor.unsqueeze(0)
        embedded_action_tensor_batch = action_embedder(action_embedder_batch)
        embedded_action_tensor = embedded_action_tensor_batch.squeeze()
        assert isinstance(embedded_action_tensor, torch.Tensor)
        assert embedded_action_tensor.shape == torch.Size([SE])
