# pylint: disable-all
import pytest
import torch

from recommender.engines.rl.ml_components.actor import Actor
from recommender.engines.rl.ml_components.history_embedder import (
    MLPHistoryEmbedder,
)


@pytest.fixture
def params():
    N = 5
    SE = 16
    BATCH_SIZE = 64
    max_N = 20
    UE = 50
    K = 3
    I = 100

    return N, SE, BATCH_SIZE, max_N, UE, K, I


def test_actor_proper_shape(params):
    N, SE, BATCH_SIZE, max_N, UE, K, I = params

    example_user = torch.rand(BATCH_SIZE, UE)
    example_services_history = torch.rand(BATCH_SIZE, N, SE)
    example_mask = torch.ones(BATCH_SIZE, I)
    example_state = (example_user, example_services_history, example_mask)

    actor = Actor(K, SE, UE, I, MLPHistoryEmbedder(SE, max_N))
    actor_output = actor(example_state)

    assert actor_output.shape == torch.Size([BATCH_SIZE, K, SE])


def test_actor_min_max(params):
    N, SE, BATCH_SIZE, max_N, UE, K, I = params

    example_user = torch.rand(BATCH_SIZE, UE)
    example_services_history = torch.rand(BATCH_SIZE, N, SE)
    example_mask = torch.ones(BATCH_SIZE, I)
    example_state = (example_user, example_services_history, example_mask)

    act_boundaries = [(-1, 1), (8.0, 10.0), (-5, -4), (0.75, 1.25)]

    for act_min, act_max in act_boundaries:
        actor = Actor(
            K,
            SE,
            UE,
            I,
            MLPHistoryEmbedder(SE, max_N),
            act_max=act_max,
            act_min=act_min,
        )
        assert (actor.act_min < actor(example_state)).all()
        assert (actor.act_max > actor(example_state)).all()
