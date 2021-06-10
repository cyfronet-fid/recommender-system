# pylint: disable-all

import torch

from recommender.engine.agents.rl_agent.models.actor import Actor
from recommender.engine.agents.rl_agent.models.history_embedder import HistoryEmbedder
from recommender.engine.agents.rl_agent.models.search_phrase_embedder import (
    SearchPhraseEmbedder,
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

    actor = Actor(K, SE, UE, I, HistoryEmbedder(SE))

    assert actor(example_state).shape == torch.Size([BATCH_SIZE, K, SE])
