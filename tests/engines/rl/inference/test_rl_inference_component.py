# pylint: disable-all
import random

import pytest

from recommender import User
from recommender.engines.panel_id_to_services_number_mapping import PANEL_ID_TO_K
from recommender.engines.rl.inference.rl_inference_component import RLInferenceComponent
from recommender.errors import UserCannotBeIdentified
from recommender.models import Service


@pytest.mark.parametrize("ver", ["v1", "v2"])
@pytest.mark.parametrize("user_id_type", ["user_id", "aai_uid"])
def test_known_user(
    mongo, generate_users_and_services, mock_rl_pipeline_exec, ver, user_id_type
):
    K = PANEL_ID_TO_K[ver]
    inference_component = RLInferenceComponent(K=K)
    assert isinstance(inference_component, RLInferenceComponent)
    assert isinstance(inference_component.engine_name, str)

    user = random.choice(list(User.objects))
    elastic_services = [service.id for service in Service.objects]

    context = {
        "user_id": {
            "panel_id": ver,
            "elastic_services": elastic_services,
            "search_data": {},
            "user_id": user.id,
        },
        "aai_uid": {
            "panel_id": ver,
            "elastic_services": elastic_services,
            "search_data": {},
            "aai_uid": user.aai_uid,
        },
    }[user_id_type]

    services_ids_1 = inference_component(context)

    assert isinstance(services_ids_1, list)
    assert len(services_ids_1) == PANEL_ID_TO_K.get(context["panel_id"])
    assert all([isinstance(service_id, int) for service_id in services_ids_1])
    assert all(service in elastic_services for service in services_ids_1)

    services_ids_2 = inference_component(context)
    assert services_ids_1 == services_ids_2


def test_user_cannot_be_identified(
    mongo, generate_users_and_services, mock_rl_pipeline_exec
):
    rl_inference_component = RLInferenceComponent(3)
    elastic_services = [service.id for service in Service.objects]

    context = {
        "panel_id": "v1",
        "elastic_services": elastic_services,
        "search_data": {},
    }

    with pytest.raises(UserCannotBeIdentified):
        rl_inference_component(context)


def _check_proper(services, ver):
    assert isinstance(services, list)
    assert len(services) == PANEL_ID_TO_K.get(ver)
    assert all([isinstance(service_id, int) for service_id in services])
