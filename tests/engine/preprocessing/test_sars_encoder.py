# pylint: disable-all

import torch

from recommender.engine.agents.rl_agent.preprocessing.state_encoder import StateEncoder
from recommender.engine.models.autoencoders import AutoEncoder
from recommender.engines.autoencoders.ml_components.embedder import Embedder
from recommender.engines.autoencoders.training.data_preparation_step import (
    precalc_users_and_service_tensors,
)
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
    user_autoencoder = AutoEncoder(features_dim=UOH, embedding_dim=UE)
    user_embedder = Embedder(user_autoencoder)

    # Service Embedder
    service_autoencoder = AutoEncoder(features_dim=SOH, embedding_dim=SE)
    service_embedder = Embedder(service_autoencoder)

    # StateEncoder
    state_encoder = StateEncoder(
        user_embedder=user_embedder,
        service_embedder=service_embedder,
    )

    sars_encoder = SarsEncoder(
        user_embedder=user_embedder, service_embedder=service_embedder
    )

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
