# pylint: disable-all

import torch

from recommender.engines.rl.ml_components.models.actor import Actor
from recommender.engines.rl.ml_components.models.history_embedder import (
    MLPHistoryEmbedder,
)


def test_actor_proper_shape():
    N = 5
    SE = 16
    BATCH_SIZE = 64
    max_N = 20
    UE = 50
    K = 3
    I = 100

    example_user = torch.rand(BATCH_SIZE, UE)
    example_services_history = torch.rand(BATCH_SIZE, N, SE)
    example_mask = torch.ones(BATCH_SIZE, I)

    example_state = (example_user, example_services_history, example_mask)

    actor = Actor(K, SE, UE, I, MLPHistoryEmbedder(SE, max_N))

    assert actor(example_state).shape == torch.Size([BATCH_SIZE, K, SE])
