# pylint: disable-all

import torch

from recommender.engine.rl_agent.models.lstm_model import LSTMModel


def test_lstm_model_proper_shape():
    input_size = 64  # service embedding dimension
    temporal_dim = 10  # length of a sequence
    batch_size = 32

    example_tensor = torch.rand(batch_size, temporal_dim, input_size)
    lstm_model = LSTMModel(input_size)

    assert lstm_model(example_tensor).shape == torch.Size([batch_size, input_size])
