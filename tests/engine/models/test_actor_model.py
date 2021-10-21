# pylint: disable-all

import torch

from recommender.engines.rl.ml_components.actor import Actor
from recommender.engines.rl.ml_components.history_embedder import (
    MLPHistoryEmbedder,
)


def test_actor_proper_shape():
    N = 5
    SE = 16
    BATCH_SIZE = 64
    UE = 50
    K = 3
    I = 100

    example_user = torch.rand(BATCH_SIZE, UE)
    example_services_history = torch.rand(BATCH_SIZE, N, SE)
    example_mask = torch.ones(BATCH_SIZE, I)

    example_state = (example_user, example_services_history, example_mask)

    actor = Actor(K, SE, UE, I, MLPHistoryEmbedder(SE))

    assert actor(example_state).shape == torch.Size([BATCH_SIZE, K, SE])
