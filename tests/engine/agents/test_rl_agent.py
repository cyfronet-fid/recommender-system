# pylint: disable-all

import pytest

from recommender.engine.agents.panel_id_to_services_number_mapping import PANEL_ID_TO_K
from recommender.engine.agents.rl_agent.preprocessing.search_data_encoder import (
    SearchDataEncoder,
)
from recommender.engine.utils import save_module
from recommender.models import Service
from recommender.engine.agents.rl_agent.rl_agent import RLAgent
from recommender.engine.agents.rl_agent.models.actor import Actor, ACTOR_V1, ACTOR_V2
from recommender.engine.agents.rl_agent.service_selector import ServiceSelector
from recommender.engine.agents.rl_agent.preprocessing.state_encoder import StateEncoder
from recommender.engine.models.autoencoders import (
    UserAutoEncoder,
    ServiceAutoEncoder,
    USERS_AUTOENCODER,
    SERVICES_AUTOENCODER,
)
from recommender.engine.agents.pre_agent.training.common import create_embedder
from recommender.engine.agents.rl_agent.models.history_embedder import (
    MLPHistoryEmbedder,
    MLP_HISTORY_EMBEDDER_V1,
    MLP_HISTORY_EMBEDDER_V2,
)
from recommender.engine.preprocessing import precalc_users_and_service_tensors
from recommender.models import User
from tests.factories.populate_database import populate_users_and_services
from recommender.utils import gen_json_dict


def test_rl_agent_call(mongo):
    populate_users_and_services(
        common_services_number=9,
        no_one_services_number=100,
        users_number=5,
        k_common_services_min=5,
        k_common_services_max=7,
    )

    precalc_users_and_service_tensors()

    UOH = len(User.objects[0].one_hot_tensor)
    UE = 32

    SOH = len(Service.objects[0].one_hot_tensor)
    SE = 64

    user_autoencoder = UserAutoEncoder(features_dim=UOH, embedding_dim=UE)
    user_embedder = create_embedder(user_autoencoder)
    save_module(module=user_autoencoder, name=USERS_AUTOENCODER)

    service_auto_encoder = ServiceAutoEncoder(features_dim=SOH, embedding_dim=SE)
    service_embedder = create_embedder(service_auto_encoder)
    save_module(module=service_auto_encoder, name=SERVICES_AUTOENCODER)

    I = len(Service.objects)

    actor_v1_history_embedder = MLPHistoryEmbedder(SE=SE)
    save_module(module=actor_v1_history_embedder, name=MLP_HISTORY_EMBEDDER_V1)

    actor_v1 = Actor(
        K=3,
        SE=SE,
        UE=UE,
        I=I,
        history_embedder=actor_v1_history_embedder,
    )
    save_module(module=actor_v1, name=ACTOR_V1)

    actor_v2_history_embedder = MLPHistoryEmbedder(SE=SE)
    save_module(module=actor_v2_history_embedder, name=MLP_HISTORY_EMBEDDER_V2)

    actor_v2 = Actor(
        K=2,
        SE=SE,
        UE=UE,
        I=I,
        history_embedder=actor_v2_history_embedder,
    )
    save_module(module=actor_v2, name=ACTOR_V2)

    search_data_encoder = SearchDataEncoder()

    state_encoder = StateEncoder(
        user_embedder=user_embedder,
        service_embedder=service_embedder,
        search_data_encoder=search_data_encoder,
    )

    service_selector = ServiceSelector(service_embedder=service_embedder)

    rl_agent_in_ram = RLAgent(
        actor_v1=actor_v1,
        actor_v2=actor_v2,
        state_encoder=state_encoder,
        service_selector=service_selector,
    )

    rl_agent_from_db = RLAgent()

    for rl_agent, kind in zip(
        (rl_agent_in_ram, rl_agent_from_db), ("in_ram", "from_db")
    ):
        for panel_id in list(PANEL_ID_TO_K.keys()):
            for anonymous_user in (True, False):
                json_dict = gen_json_dict(
                    panel_id=panel_id, anonymous_user=anonymous_user
                )
                recommended_services_ids = rl_agent.call(json_dict)

                assert type(recommended_services_ids) == list
                assert len(recommended_services_ids) == PANEL_ID_TO_K[panel_id]

                # For anonymous user recommendations should change over time (because they are random)
                # but for non anonymous user they should stay same over time.
                recommendations = [rl_agent.call(json_dict) for _ in range(10)]

                assert (
                    all([recommendations[0] == r for r in recommendations])
                    != anonymous_user
                )
