# pylint: disable-all

import torch

from recommender.engine.agents.rl_agent.services2weights import Services2Weights
from recommender.engine.agents.rl_agent.preprocessing.reward_encoder import (
    RewardEncoder,
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
from recommender.engine.preprocessing import precalc_users_and_service_tensors
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
from tests.factories.sars import SarsFactory
from tests.factories.search_data import SearchDataFactory
from tests.factories.state import StateFactory


def test_sars_encoder(mongo):
    B = 3
    SARSes_K_2 = SarsFactory.create_batch(
        B,
        state=StateFactory(search_data=SearchDataFactory(q=None)),
        next_state=StateFactory(search_data=SearchDataFactory(q=None)),
        K_2=True,
    )
    SARSes_K_3 = SarsFactory.create_batch(
        B,
        state=StateFactory(search_data=SearchDataFactory(q=None)),
        next_state=StateFactory(search_data=SearchDataFactory(q=None)),
        K_3=True,
    )

    # Generate data
    precalc_users_and_service_tensors()
    for SARS_K_2 in SARSes_K_2:
        SARS_K_2.reload()
    for SARS_K_3 in SARSes_K_3:
        SARS_K_3.reload()

    # Constants
    UOH = len(User.objects.first().one_hot_tensor)
    UE = 32

    SOH = len(Service.objects.first().one_hot_tensor)
    SE = 64
    I = len(Service.objects)

    # User Embedder
    user_autoencoder = UserAutoEncoder(features_dim=UOH, embedding_dim=UE)
    user_embedder = create_embedder(user_autoencoder)
    save_module(module=user_autoencoder, name=USERS_AUTOENCODER)

    # Service Embedder
    service_auto_encoder = ServiceAutoEncoder(features_dim=SOH, embedding_dim=SE)
    service_embedder = create_embedder(service_auto_encoder)
    save_module(module=service_auto_encoder, name=SERVICES_AUTOENCODER)

    # StateEncoder
    state_encoder = StateEncoder(
        user_embedder=user_embedder,
        service_embedder=service_embedder,
        search_data_encoder=SearchDataEncoder(),
    )

    services2weights = Services2Weights(service_embedder=service_embedder)

    # Reward Encoder
    reward_encoder = RewardEncoder()

    sars_encoder_in_ram = SarsEncoder(
        state_encoder=state_encoder,
        reward_encoder=reward_encoder,
        services2weights=services2weights,
    )

    sars_encoder_from_db = SarsEncoder()

    for sars_encoder in (sars_encoder_in_ram, sars_encoder_from_db):
        for K, SARSes_K in zip((2, 3), (SARSes_K_2, SARSes_K_3)):
            batch = sars_encoder(SARSes_K)

            assert batch[STATE][USER].shape == torch.Size([B, UE])
            assert batch[STATE][SERVICES_HISTORY].shape[0] == B
            assert batch[STATE][SERVICES_HISTORY].shape[2] == SE
            assert batch[STATE][MASK].shape == torch.Size([B, I])

            assert batch[ACTION].shape == torch.Size([B, K, SE])
            assert batch[REWARD].shape == torch.Size([B])

            assert batch[NEXT_STATE][USER].shape == torch.Size([B, UE])
            assert batch[NEXT_STATE][SERVICES_HISTORY].shape[0] == B
            assert batch[NEXT_STATE][SERVICES_HISTORY].shape[2] == SE
            assert batch[NEXT_STATE][MASK].shape == torch.Size([B, I])
