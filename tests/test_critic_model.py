# pylint: disable-all

import torch

from recommender.engine.agents.rl_agent.models.critic import CriticModel
from recommender.engine.agents.rl_agent.models.history_embedder import HistoryEmbedder
from recommender.engine.agents.rl_agent.models.search_phrase_embedder import (
    SearchPhraseEmbedder,
)


def test_critic_model():
    batch_size = 32

    K = 3
    SE = 128
    UE = 32
    SPE = 100

    X = 15
    N = 25

    user = torch.rand(batch_size, UE)
    services_history = torch.rand(batch_size, N, SE)
    filters = torch.rand(batch_size, SE)
    search_phrase = torch.rand(batch_size, X, SPE)
    state = (user, services_history, filters, search_phrase)

    action = torch.rand(batch_size, K, SE)

    search_phrase_embedder = SearchPhraseEmbedder(SPE)
    history_embedder = HistoryEmbedder(SE)
    action_embedder = None

    critic_model = CriticModel(
        K, SE, UE, SPE, search_phrase_embedder, history_embedder, action_embedder
    )

    action_value = critic_model(state, action)

    assert isinstance(action_value, torch.Tensor)
    assert action_value.shape == torch.Size([batch_size, 1])

    action_value[0].backward()
