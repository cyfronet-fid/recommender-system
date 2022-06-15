# pylint: disable-all

import torch

from recommender.engines.nlp_embedders.embedders import (
    Users2tensorsEmbedder,
    Services2tensorsEmbedder,
)
from recommender.engines.rl.ml_components.services2weights import Services2Weights
from recommender.engines.rl.ml_components.critic import Critic
from recommender.engines.rl.ml_components.history_embedder import (
    MLPHistoryEmbedder,
)
from recommender.engines.rl.ml_components.state_encoder import StateEncoder
from recommender.models import User
from recommender.models import Service
from tests.factories.search_data import SearchDataFactory
from tests.factories.state import StateFactory


def test_critic(mongo):
    # Generate data
    state = StateFactory(search_data=SearchDataFactory(q=None), non_empty_history=True)
    action_v1 = Service.objects[:3]
    action_v2 = Service.objects[:2]

    # Constants
    UE = Users2tensorsEmbedder().embedding_dim
    SE = Services2tensorsEmbedder().embedding_dim

    I = len(Service.objects)
    max_N = 20

    # HistoryEmbedder v1
    history_embedder_v1 = MLPHistoryEmbedder(SE=SE, max_N=max_N)

    # HistoryEmbedder v2
    history_embedder_v2 = MLPHistoryEmbedder(SE=SE, max_N=max_N)

    # State encoder
    state_encoder = StateEncoder()

    # critic
    critic_v1_in_ram = Critic(
        K=3, SE=SE, UE=UE, I=I, history_embedder=history_embedder_v1
    )

    critic_v2_in_ram = Critic(
        K=2, SE=SE, UE=UE, I=I, history_embedder=history_embedder_v2
    )

    # TODO: refactor
    critics_v1 = (critic_v1_in_ram,)
    critics_v2 = (critic_v2_in_ram,)
    critics = critics_v1 + critics_v2

    services2weights = Services2Weights()

    batch_size = 64
    state_tensors_batch = state_encoder([state] * batch_size)
    weights_tensor_batch_v1 = services2weights([[s.id for s in action_v1]] * batch_size)
    weights_tensor_batch_v2 = services2weights([[s.id for s in action_v2]] * batch_size)

    for critic in critics:
        weights_tensor_batch = (
            weights_tensor_batch_v1 if critic in critics_v1 else weights_tensor_batch_v2
        )
        action_value_batch = critic(
            state=state_tensors_batch, action=weights_tensor_batch
        )
        assert isinstance(action_value_batch, torch.Tensor)
        assert action_value_batch.shape == torch.Size([batch_size, 1])

        action_value_batch[0].backward()
