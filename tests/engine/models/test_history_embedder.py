# pylint: disable-all
import random

import pytest
import torch

from recommender.engines.rl.ml_components.models.history_embedder import (
    LSTMHistoryEmbedder,
    MLPHistoryEmbedder,
)


@pytest.fixture
def params():
    batch_size = 64
    max_N = random.randint(5, 50)  # length of an output sequence
    SE = 128  # service embedding dimension
    return batch_size, max_N, SE


def test_lstm_history_embedder_proper_shape(params):
    batch_size, _, SE = params
    example_history = torch.rand(batch_size, 25, SE)
    history_embedder = LSTMHistoryEmbedder(SE)

    assert history_embedder(example_history).shape == torch.Size([batch_size, SE])


def test_mlp_history_embedder_proper_shape(params):
    batch_size, max_N, SE = params
    history_embedder = MLPHistoryEmbedder(SE, max_N)

    example_history = torch.rand(batch_size, random.randint(1, max_N), SE)
    assert history_embedder(example_history).shape == torch.Size([batch_size, SE])

    example_history = torch.rand(batch_size, random.randint(max_N, 1000), SE)
    assert history_embedder(example_history).shape == torch.Size([batch_size, SE])


def test_mlp_history_embedder_align_history(params):
    batch_size, max_N, SE = params
    history_embedder = MLPHistoryEmbedder(SE, max_N)

    history = torch.rand(batch_size, random.randint(1, max_N), SE)
    assert history_embedder._align_history(history).shape == torch.Size(
        [batch_size, max_N, SE]
    )

    history = torch.rand(batch_size, random.randint(max_N, 1000), SE)
    assert history_embedder._align_history(history).shape == torch.Size(
        [batch_size, max_N, SE]
    )
