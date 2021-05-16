# pylint: disable-all

import torch

from recommender.engine.agents.rl_agent.models.search_phrase_embedder import (
    SearchPhraseEmbedder,
)


def test_search_phrase_embedder_proper_shape():
    SPE = 64  # search phrase embedding dimension
    X = 10  # number of subwords
    batch_size = 32

    example_tensor = torch.rand(batch_size, X, SPE)
    search_phrase_model = SearchPhraseEmbedder(SPE)

    assert search_phrase_model(example_tensor).shape == torch.Size([batch_size, SPE])
