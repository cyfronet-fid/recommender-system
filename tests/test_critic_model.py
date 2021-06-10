# pylint: disable-all

import torch

from recommender.engine.agents.rl_agent.models.critic import Critic
from recommender.engine.agents.rl_agent.models.history_embedder import (
    HistoryEmbedder,
    HISTORY_EMBEDDER_V1,
    HISTORY_EMBEDDER_V2,
)
from recommender.engine.agents.rl_agent.models.search_phrase_embedder import (
    SearchPhraseEmbedder,
    SEARCH_PHRASE_EMBEDDER_V1,
    SEARCH_PHRASE_EMBEDDER_V2,
)
from recommender.engine.agents.rl_agent.preprocessing.filters_encoder import (
    FiltersEncoder,
)
from recommender.engine.agents.rl_agent.preprocessing.searchphrase_encoder import (
    SearchPhraseEncoder,
)
from recommender.engine.agents.rl_agent.preprocessing.state_encoder import (
    StateEncoder,
    MaskEncoder,
)
from recommender.engine.models.autoencoders import (
    UserAutoEncoder,
    create_embedder,
    USERS_AUTOENCODER,
    ServiceAutoEncoder,
    SERVICES_AUTOENCODER,
)
from recommender.engine.preprocessing import load_last_transformer, SERVICES
from recommender.engine.utils import save_module
from recommender.models import User
from recommender.engine.agents.rl_agent.preprocessing.action_encoder import (
    ActionEncoder,
)
from recommender.engine.preprocessing import precalc_users_and_service_tensors
from recommender.models import Service
from tests.factories.state import StateFactory


def test_critic(mongo):
    # Generate data
    state = StateFactory()
    precalc_users_and_service_tensors()
    state.reload()
    action = Service.objects[:3]

    # Constants
    UOH = len(User.objects.first().tensor)
    UE = 32

    SOH = len(Service.objects.first().tensor)
    SE = 64

    SPE = 100

    # User Embedder
    user_autoencoder = UserAutoEncoder(features_dim=UOH, embedding_dim=UE)
    user_embedder = create_embedder(user_autoencoder)
    save_module(module=user_autoencoder, name=USERS_AUTOENCODER)

    # Service Embedder
    service_auto_encoder = ServiceAutoEncoder(features_dim=SOH, embedding_dim=SE)
    service_embedder = create_embedder(service_auto_encoder)
    save_module(module=service_auto_encoder, name=SERVICES_AUTOENCODER)

    # SearchPhraseEmbedder v1
    search_phrase_embedder_v1 = SearchPhraseEmbedder(SPE=SPE, num_layers=3, dropout=0.5)
    save_module(module=search_phrase_embedder_v1, name=SEARCH_PHRASE_EMBEDDER_V1)

    # HistoryEmbedder v1
    history_embedder_v1 = HistoryEmbedder(SE=SE, num_layers=3, dropout=0.5)
    save_module(module=history_embedder_v1, name=HISTORY_EMBEDDER_V1)

    # SearchPhraseEmbedder v2
    search_phrase_embedder_v2 = SearchPhraseEmbedder(SPE=SPE, num_layers=3, dropout=0.5)
    save_module(module=search_phrase_embedder_v2, name=SEARCH_PHRASE_EMBEDDER_V2)

    # HistoryEmbedder v2
    history_embedder_v2 = HistoryEmbedder(SE=SE, num_layers=3, dropout=0.5)
    save_module(module=history_embedder_v2, name=HISTORY_EMBEDDER_V2)

    # StateEncoder
    mask_encoder = MaskEncoder()

    state_encoder = StateEncoder(
        user_embedder=user_embedder,
        service_embedder=service_embedder,
        mask_encoder=mask_encoder,
    )

    # ActionEncoder
    action_encoder = ActionEncoder(service_embedder=service_embedder)

    # critic
    critic_v1_in_ram = Critic(
        K=3,
        SE=SE,
        UE=UE,
        SPE=SPE,
        search_phrase_embedder=search_phrase_embedder_v1,
        history_embedder=history_embedder_v1,
    )

    critic_v1_from_db = Critic(
        K=3,
        SE=SE,
        UE=UE,
        SPE=SPE,
        search_phrase_embedder=search_phrase_embedder_v1,
        history_embedder=history_embedder_v1,
    )

    critic_v2_in_ram = Critic(
        K=2,
        SE=SE,
        UE=UE,
        SPE=SPE,
        search_phrase_embedder=search_phrase_embedder_v2,
        history_embedder=history_embedder_v2,
    )

    critic_v2_from_db = Critic(
        K=2,
        SE=SE,
        UE=UE,
        SPE=SPE,
    )

    critics = (critic_v1_in_ram, critic_v1_from_db, critic_v2_in_ram, critic_v2_from_db)

    for batch_size in (1, 64):
        # Input
        state_tensors_batch = state_encoder([state])

        action_tensor = action_encoder(action)
        action_tensor_batch = torch.stack([action_tensor] * batch_size)

        for critic in critics:
            action_value_batch = critic(
                state=state_tensors_batch, action=action_tensor_batch
            )
            assert isinstance(action_value_batch, torch.Tensor)
            assert action_value_batch.shape == torch.Size([batch_size, 1])

            action_value_batch[0].backward()
