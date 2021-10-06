# pylint: disable-all

import torch

from recommender.engine.agents.rl_agent.models.history_embedder import (
    LSTMHistoryEmbedder,
    MLPHistoryEmbedder,
)


def test_lstm_history_embedder_proper_shape():
    SE = 64  # service embedding dimension
    N = 10  # length of a sequence
    batch_size = 32

    example_history = torch.rand(batch_size, N, SE)
    history_embedder = LSTMHistoryEmbedder(SE)

    assert history_embedder(example_history).shape == torch.Size([batch_size, SE])


def test_mlp_history_embedder_proper_shape():
    SE = 64  # service embedding dimension
    N = 10  # length of a sequence
    batch_size = 32

    example_history = torch.rand(batch_size, N, SE)
    history_embedder = MLPHistoryEmbedder(SE, N)

    assert history_embedder(example_history).shape == torch.Size([batch_size, SE])
