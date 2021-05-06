# pylint: disable-all

import torch

from recommender.engine.rl_agent.models.actor_model import ActorModel
from recommender.engine.rl_agent.models.history_model import HistoryModel
from recommender.engine.rl_agent.models.search_phrase_model import SearchPhraseModel


def test_actor_model_proper_shape():
    N = 5
    SE = 16
    BATCH_SIZE = 64
    UE = 50
    SPE = 100
    X = 15
    K = 3

    example_user = torch.rand(BATCH_SIZE, UE)
    example_services_history = torch.rand(BATCH_SIZE, N, SE)
    example_filters = torch.rand(BATCH_SIZE, SE)
    example_search_phrase = torch.rand(BATCH_SIZE, X, SPE)

    actor_model = ActorModel(K, SE, UE, SPE, SearchPhraseModel(SPE), HistoryModel(SE))

    assert actor_model(
        example_user, example_services_history, example_filters, example_search_phrase
    ).shape == torch.Size([BATCH_SIZE, K, SE])
