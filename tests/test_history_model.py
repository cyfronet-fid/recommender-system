# pylint: disable-all

import torch

from recommender.engine.rl_agent.models.history_model import HistoryModel


def test_history_model_proper_shape():
    SE = 64  # service embedding dimension
    N = 10  # length of a sequence
    batch_size = 32

    example_tensor = torch.rand(batch_size, N, SE)
    history_model = HistoryModel(SE)

    assert history_model(example_tensor).shape == torch.Size([batch_size, SE])
