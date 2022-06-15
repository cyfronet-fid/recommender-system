# pylint: disable-all
import pytest
import torch

from recommender.engines.nlp_embedders.embedders import (
    Users2tensorsEmbedder,
    Services2tensorsEmbedder,
)
from recommender.engines.rl.ml_components.state_encoder import StateEncoder
from recommender.models import User, Service
from recommender.engines.rl.ml_components.sars_encoder import (
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


@pytest.mark.skip(reason="TODO")
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

    # Constants
    UE = Users2tensorsEmbedder().embedding_dim
    SE = Services2tensorsEmbedder().embedding_dim
    I = len(Service.objects)

    # StateEncoder
    state_encoder = StateEncoder()

    sars_encoder = SarsEncoder()

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
