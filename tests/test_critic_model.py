# pylint: disable-all

import torch

from recommender.engine.agents.rl_agent.action_inverter import ActionInverter
from recommender.engine.agents.rl_agent.models.action_embedder import (
    ActionEmbedder,
    ACTION_EMBEDDER_V1,
    ACTION_EMBEDDER_V2,
)
from recommender.engine.agents.rl_agent.models.critic import Critic
from recommender.engine.agents.rl_agent.models.history_embedder import (
    HistoryEmbedder,
    HISTORY_EMBEDDER_V1,
    HISTORY_EMBEDDER_V2,
)
from recommender.engine.agents.rl_agent.preprocessing.search_data_encoder import (
    SearchDataEncoder,
)
from recommender.engine.agents.rl_agent.preprocessing.state_encoder import StateEncoder
from recommender.engine.models.autoencoders import (
    UserAutoEncoder,
    create_embedder,
    USERS_AUTOENCODER,
    ServiceAutoEncoder,
    SERVICES_AUTOENCODER,
)
from recommender.engine.utils import save_module
from recommender.models import User
from recommender.engine.preprocessing import precalc_users_and_service_tensors
from recommender.models import Service
from tests.factories.search_data import SearchDataFactory
from tests.factories.state import StateFactory


def test_critic(mongo):
    # Generate data
    state = StateFactory(search_data=SearchDataFactory(q=None))
    precalc_users_and_service_tensors()
    state.reload()
    action = Service.objects[:3]

    # Constants
    UOH = len(User.objects.first().tensor)
    UE = 32

    SOH = len(Service.objects.first().tensor)
    SE = 64

    # User Embedder
    user_auto_encoder = UserAutoEncoder(features_dim=UOH, embedding_dim=UE)
    user_embedder = create_embedder(user_auto_encoder)
    save_module(module=user_auto_encoder, name=USERS_AUTOENCODER)

    # Service Embedder
    service_auto_encoder = ServiceAutoEncoder(features_dim=SOH, embedding_dim=SE)
    service_embedder = create_embedder(service_auto_encoder)
    save_module(module=service_auto_encoder, name=SERVICES_AUTOENCODER)

    # HistoryEmbedder v1
    history_embedder_v1 = HistoryEmbedder(SE=SE, num_layers=3, dropout=0.5)
    save_module(module=history_embedder_v1, name=HISTORY_EMBEDDER_V1)

    # HistoryEmbedder v2
    history_embedder_v2 = HistoryEmbedder(SE=SE, num_layers=3, dropout=0.5)
    save_module(module=history_embedder_v2, name=HISTORY_EMBEDDER_V2)

    # Search data encoder
    search_data_encoder = SearchDataEncoder()

    # State encoder
    state_encoder = StateEncoder(
        user_embedder=user_embedder,
        service_embedder=service_embedder,
        search_data_encoder=search_data_encoder,
    )

    # ActionEmbedder v1
    action_embedder_v1 = ActionEmbedder(SE=SE)
    save_module(module=action_embedder_v1, name=ACTION_EMBEDDER_V1)

    # ActionEmbedder v2
    action_embedder_v2 = ActionEmbedder(SE=SE)
    save_module(module=action_embedder_v2, name=ACTION_EMBEDDER_V2)

    # critic
    critic_v1_in_ram = Critic(
        K=3,
        SE=SE,
        UE=UE,
        I=len(Service.objects),
        history_embedder=history_embedder_v1,
        action_embedder=action_embedder_v1,
    )

    critic_v1_from_db = Critic(K=3, SE=SE, UE=UE, I=len(Service.objects))

    critic_v2_in_ram = Critic(
        K=2,
        SE=SE,
        UE=UE,
        I=len(Service.objects),
        history_embedder=history_embedder_v2,
        action_embedder=action_embedder_v2,
    )

    critic_v2_from_db = Critic(K=2, SE=SE, UE=UE, I=len(Service.objects))

    critics = (critic_v1_in_ram, critic_v1_from_db, critic_v2_in_ram, critic_v2_from_db)

    action_inverter = ActionInverter(service_embedder=service_embedder)

    batch_size = 64
    state_tensors_batch = state_encoder([state] * batch_size)
    action_tensor_batch = action_inverter([[s.id for s in action]] * batch_size)

    for critic in critics:
        action_value_batch = critic(
            state=state_tensors_batch, action=action_tensor_batch
        )
        assert isinstance(action_value_batch, torch.Tensor)
        assert action_value_batch.shape == torch.Size([batch_size, 1])

        action_value_batch[0].backward()
