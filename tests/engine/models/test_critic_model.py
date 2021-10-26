# pylint: disable-all

import torch

from recommender.engines.autoencoders.ml_components.autoencoder import AutoEncoder
from recommender.engines.rl.ml_components.services2weights import Services2Weights
from recommender.engines.rl.ml_components.critic import Critic
from recommender.engines.rl.ml_components.history_embedder import (
    MLPHistoryEmbedder,
)
from recommender.engines.autoencoders.ml_components.embedder import Embedder
from recommender.engines.rl.ml_components.state_encoder import StateEncoder
from recommender.models import User
from recommender.engines.autoencoders.training.data_preparation_step import (
    precalc_users_and_service_tensors,
)
from recommender.models import Service
from tests.factories.search_data import SearchDataFactory
from tests.factories.state import StateFactory


def test_critic(mongo):
    # Generate data
    state = StateFactory(search_data=SearchDataFactory(q=None), non_empty_history=True)
    precalc_users_and_service_tensors()
    state.reload()
    action_v1 = Service.objects[:3]
    action_v2 = Service.objects[:2]

    # Constants
    UOH = len(User.objects.first().one_hot_tensor)
    UE = 32

    SOH = len(Service.objects.first().one_hot_tensor)
    SE = 64

    I = len(Service.objects)

    # User Embedder
    user_autoencoder = AutoEncoder(features_dim=UOH, embedding_dim=UE)
    user_embedder = Embedder(user_autoencoder)

    # Service Embedder
    service_autoencoder = AutoEncoder(features_dim=SOH, embedding_dim=SE)
    service_embedder = Embedder(service_autoencoder)

    # HistoryEmbedder v1
    history_embedder_v1 = MLPHistoryEmbedder(SE=SE)

    # HistoryEmbedder v2
    history_embedder_v2 = MLPHistoryEmbedder(SE=SE)

    # State encoder
    state_encoder = StateEncoder(
        user_embedder=user_embedder,
        service_embedder=service_embedder,
        use_cached_embeddings=False,
    )

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

    services2weights = Services2Weights(
        service_embedder=service_embedder, use_cached_embeddings=False
    )

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
