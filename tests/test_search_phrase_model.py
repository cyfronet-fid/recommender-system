# pylint: disable-all

import torch

from recommender.engine.rl_agent.models.search_phrase_model import SearchPhraseModel


def test_search_phrase_model_proper_shape():
    SPE = 64  # search phrase embedding dimension
    X = 10  # number of subwords
    batch_size = 32

    example_tensor = torch.rand(batch_size, X, SPE)
    search_phrase_model = SearchPhraseModel(SPE)

    assert search_phrase_model(example_tensor).shape == torch.Size([batch_size, SPE])
