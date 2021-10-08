# pylint: disable=no-member, wrong-import-position, fixme

"""Fake data training"""

from dotenv import load_dotenv

load_dotenv()

from mongoengine import connect, disconnect
from settings import TestingConfig, DevelopmentConfig

from recommender.engine.agents.rl_agent.models.actor import Actor, ACTOR_V1, ACTOR_V2
from recommender.engine.agents.rl_agent.models.history_embedder import (
    MLPHistoryEmbedder,
    MLP_HISTORY_EMBEDDER_V1,
    MLP_HISTORY_EMBEDDER_V2,
)

from recommender.engine.models.autoencoders import (
    UserAutoEncoder,
    create_embedder,
    USERS_AUTOENCODER,
    ServiceAutoEncoder,
    SERVICES_AUTOENCODER,
)
from recommender.engine.utils import save_module
from recommender.models import User, Service
from recommender.engine.preprocessing import (
    precalc_users_and_service_tensors,
    load_last_transformer,
    SERVICES,
)
from recommender.engine.preprocessing import USERS, save_transformer
from tests.factories.populate_database import populate_users_and_services


if __name__ == "__main__":
    connect(host=TestingConfig.MONGODB_HOST)

    populate_users_and_services(
        common_services_number=9,
        no_one_services_number=9,
        users_number=5,
        k_common_services_min=5,
        k_common_services_max=7,
    )

    precalc_users_and_service_tensors()

    user_transformer = load_last_transformer(USERS)
    service_transformer = load_last_transformer(SERVICES)

    UOH = len(User.objects[0].one_hot_tensor)
    UE = 32

    SOH = len(Service.objects[0].one_hot_tensor)
    SE = 64

    I = len(Service.objects)

    user_autoencoder = UserAutoEncoder(features_dim=UOH, embedding_dim=UE)
    # TODO: user_autoencoder training
    user_embedder = create_embedder(user_autoencoder)

    service_auto_encoder = ServiceAutoEncoder(features_dim=SOH, embedding_dim=SE)
    # TODO: service_autoencoder training
    service_embedder = create_embedder(service_auto_encoder)

    actor_v1_history_embedder = MLPHistoryEmbedder(SE=SE)

    actor_v1 = Actor(
        K=3,
        SE=SE,
        UE=UE,
        I=I,
        history_embedder=actor_v1_history_embedder,
    )
    # TODO: actor_v1 training

    actor_v2_history_embedder = MLPHistoryEmbedder(SE=SE)

    actor_v2 = Actor(
        K=2,
        SE=SE,
        UE=UE,
        I=I,
        history_embedder=actor_v2_history_embedder,
    )
    # TODO: actor_v2 training

    disconnect()
    connect(host=DevelopmentConfig.MONGODB_HOST)

    save_module(module=user_autoencoder, name=USERS_AUTOENCODER)
    save_module(module=service_auto_encoder, name=SERVICES_AUTOENCODER)
    save_module(module=actor_v1_history_embedder, name=MLP_HISTORY_EMBEDDER_V1)
    save_module(module=actor_v1, name=ACTOR_V1)
    save_module(module=actor_v2_history_embedder, name=MLP_HISTORY_EMBEDDER_V2)
    save_module(module=actor_v2, name=ACTOR_V2)

    save_transformer(user_transformer, USERS)
    save_transformer(service_transformer, SERVICES)
