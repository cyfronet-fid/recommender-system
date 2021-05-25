# pylint: disable-all

import torch

from recommender.engine.agents.rl_agent.preprocessing.reward_encoder import (
    RewardEncoder,
)
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
from recommender.engine.agents.rl_agent.preprocessing.action_encoder import (
    ActionEncoder,
)
from recommender.engine.agents.rl_agent.preprocessing.filters_encoder import (
    FiltersEncoder,
)
from recommender.engine.agents.rl_agent.preprocessing.searchphrase_encoder import (
    SearchPhraseEncoder,
)
from recommender.engine.agents.rl_agent.preprocessing.state_encoder import StateEncoder
from recommender.engine.models.autoencoders import (
    UserAutoEncoder,
    create_embedder,
    USERS_AUTOENCODER,
    ServiceAutoEncoder,
    SERVICES_AUTOENCODER,
)
from recommender.engine.preprocessing import (
    precalc_users_and_service_tensors,
    load_last_transformer,
    SERVICES,
)
from recommender.engine.utils import save_module
from recommender.models import User, Service
from recommender.engine.agents.rl_agent.preprocessing.sars_encoder import (
    SarsEncoder,
    STATE,
    USER,
    SERVICES_HISTORY,
    FILTERS,
    SEARCH_PHRASE,
    ACTION,
    REWARD,
    NEXT_STATE,
)
from tests.factories.sars import SarsFactory


def test_sars_encoder(mongo):
    SARS_K_2 = SarsFactory(K_2=True)
    SARS_K_3 = SarsFactory(K_3=True)

    # Generate data
    precalc_users_and_service_tensors()
    SARS_K_2.reload()
    SARS_K_3.reload()

    # Service transformer
    service_transformer = load_last_transformer(SERVICES)

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

    # SearchPhraseEncoder
    search_phrase_encoder = SearchPhraseEncoder(dim=SPE)

    # FiltersEncoder
    filters_encoder = FiltersEncoder(
        service_transformer=service_transformer, service_embedder=service_embedder
    )

    # StateEncoder
    state_encoder = StateEncoder(
        user_embedder=user_embedder,
        service_embedder=service_embedder,
        search_phrase_encoder=search_phrase_encoder,
        filters_encoder=filters_encoder,
    )

    # ActionEncoder
    action_encoder = ActionEncoder(service_embedder=service_embedder)

    # Reward Encoder
    reward_encoder = RewardEncoder()

    sars_encoder_in_ram = SarsEncoder(
        state_encoder=state_encoder,
        action_encoder=action_encoder,
        reward_encoder=reward_encoder,
    )

    sars_encoder_from_db = SarsEncoder()

    for sars_encoder in (sars_encoder_in_ram, sars_encoder_from_db):
        for K, SARS_K in zip((2, 3), (SARS_K_2, SARS_K_3)):
            example = sars_encoder(SARS_K)

            assert example[STATE][USER].shape == torch.Size([UE])
            assert example[STATE][SERVICES_HISTORY].shape[1] == SE
            assert example[STATE][FILTERS].shape == torch.Size([SE])
            assert example[STATE][SEARCH_PHRASE].shape[1] == SPE

            assert example[ACTION].shape == torch.Size([K, SE])
            assert example[REWARD].shape == torch.Size([])

            assert example[NEXT_STATE][USER].shape == torch.Size([UE])
            assert example[NEXT_STATE][SERVICES_HISTORY].shape[1] == SE
            assert example[NEXT_STATE][FILTERS].shape == torch.Size([SE])
            assert example[NEXT_STATE][SEARCH_PHRASE].shape[1] == SPE
