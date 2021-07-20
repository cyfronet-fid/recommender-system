# pylint: disable-all

import torch
from torch.utils.data import DataLoader

from recommender.engine.agents.rl_agent.services2weights import Services2Weights
from recommender.engine.agents.rl_agent.datasets.replay_buffer import (
    ReplayBuffer,
    collate_batch,
)
from recommender.engine.agents.rl_agent.preprocessing.search_data_encoder import (
    SearchDataEncoder,
)
from recommender.engine.agents.rl_agent.preprocessing.state_encoder import StateEncoder
from recommender.models import Sars
from tests.factories.sars import SarsFactory
from recommender.engine.agents.rl_agent.preprocessing.reward_encoder import (
    RewardEncoder,
)
from recommender.engine.agents.rl_agent.models.history_embedder import (
    HistoryEmbedder,
    HISTORY_EMBEDDER_V1,
    HISTORY_EMBEDDER_V2,
)
from recommender.engine.models.autoencoders import (
    UserAutoEncoder,
    create_embedder,
    USERS_AUTOENCODER,
    ServiceAutoEncoder,
    SERVICES_AUTOENCODER,
)
from recommender.engine.preprocessing import (
    precalc_users_and_service_tensors,
)
from recommender.engine.utils import save_module
from recommender.models import User, Service
from recommender.engine.agents.rl_agent.preprocessing.sars_encoder import (
    SarsEncoder,
    STATE,
    USER,
    SERVICES_HISTORY,
    ACTION,
    REWARD,
    NEXT_STATE,
    MASK,
)
from tests.factories.search_data import SearchDataFactory
from tests.factories.state import StateFactory


def test_replay_buffer(mongo):
    SARSes_K_2 = SarsFactory.create_batch(
        6,
        state=StateFactory(search_data=SearchDataFactory(q=None)),
        next_state=StateFactory(search_data=SearchDataFactory(q=None)),
        K_2=True,
    )
    SARSes_K_3 = SarsFactory.create_batch(
        6,
        state=StateFactory(search_data=SearchDataFactory(q=None)),
        next_state=StateFactory(search_data=SearchDataFactory(q=None)),
        K_3=True,
    )
    # Generate data
    precalc_users_and_service_tensors()

    # Constants
    UOH = len(User.objects.first().tensor)
    UE = 32

    SOH = len(Service.objects.first().tensor)
    SE = 64
    I = len(Service.objects)

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

    search_data_encoder = SearchDataEncoder()

    # StateEncoder
    state_encoder = StateEncoder(
        user_embedder=user_embedder,
        service_embedder=service_embedder,
        search_data_encoder=search_data_encoder,
    )

    services2weights = Services2Weights(service_embedder=service_embedder)

    # Reward Encoder
    reward_encoder = RewardEncoder()

    sars_encoder = SarsEncoder(
        state_encoder=state_encoder,
        reward_encoder=reward_encoder,
        services2weights=services2weights,
    )

    for K in (2, 3):
        replay_buffer = ReplayBuffer(Sars.objects, K=K, sars_encoder=sars_encoder)
        for BATCH_SIZE in (2, 3):
            replay_buffer_dl = DataLoader(
                replay_buffer,
                batch_size=BATCH_SIZE,
                shuffle=True,
                collate_fn=collate_batch,
            )

            batch = next(iter(replay_buffer_dl))

            assert batch[STATE][USER].shape == torch.Size([BATCH_SIZE, UE])
            assert batch[STATE][SERVICES_HISTORY].shape[0] == BATCH_SIZE
            assert batch[STATE][SERVICES_HISTORY].shape[2] == SE
            assert batch[STATE][MASK].shape == torch.Size([BATCH_SIZE, I])

            assert batch[ACTION].shape == torch.Size([BATCH_SIZE, K, SE])

            assert batch[REWARD].shape == torch.Size([BATCH_SIZE])

            assert batch[NEXT_STATE][USER].shape == torch.Size([BATCH_SIZE, UE])
            assert batch[NEXT_STATE][SERVICES_HISTORY].shape[0] == BATCH_SIZE
            assert batch[NEXT_STATE][SERVICES_HISTORY].shape[2] == SE
            assert batch[NEXT_STATE][MASK].shape == torch.Size([BATCH_SIZE, I])
