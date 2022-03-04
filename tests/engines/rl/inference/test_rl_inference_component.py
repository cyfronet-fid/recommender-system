# pylint: disable-all
import random

import pytest

from recommender import User
from recommender.engines.panel_id_to_services_number_mapping import PANEL_ID_TO_K
from recommender.engines.rl.inference.rl_inference_component import RLInferenceComponent
from recommender.models import Category


@pytest.mark.parametrize("ver", ["v1", "v2"])
def test_known_user(mongo, generate_users_and_services, mock_rl_pipeline_exec, ver):
    K = PANEL_ID_TO_K[ver]
    inference_component = RLInferenceComponent(K=K)

    user = random.choice(list(User.objects))
    context = {"panel_id": ver, "search_data": {}, "user_id": user.id}

    services_ids_1 = inference_component(context)

    assert isinstance(services_ids_1, list)
    assert len(services_ids_1) == PANEL_ID_TO_K.get(context["panel_id"])
    assert all([isinstance(service_id, int) for service_id in services_ids_1])

    services_ids_2 = inference_component(context)
    assert services_ids_1 == services_ids_2


@pytest.mark.parametrize("ver", ["v1", "v2"])
def test_unknown_user(mongo, generate_users_and_services, mock_rl_pipeline_exec, ver):
    K = PANEL_ID_TO_K[ver]
    inference_component = RLInferenceComponent(K=K)

    contexts = [
        {"panel_id": ver, "search_data": {}, "user_id": -1},
        {"panel_id": ver, "search_data": {}},
    ]

    for context in contexts:
        services_ids_1 = inference_component(context)
        _check_proper(services_ids_1, ver)

        services_ids_2 = inference_component(context)
        _check_proper(services_ids_2, ver)
        assert services_ids_1 != services_ids_2


@pytest.mark.parametrize("ver", ["v1", "v2"])
def test_not_empty_search_data(
    mongo, generate_users_and_services, mock_rl_pipeline_exec, ver
):
    K = PANEL_ID_TO_K[ver]
    inference_component = RLInferenceComponent(K=K)

    user = User.objects.first()
    context = {
        "panel_id": ver,
        "search_data": {"categories": list(Category.objects.distinct("id"))},
        "user_id": user.id,
    }

    services_ids_1 = inference_component(context)
    _check_proper(services_ids_1, ver)

    services_ids_2 = inference_component(context)
    _check_proper(services_ids_2, ver)
    assert services_ids_1 == services_ids_2


def _check_proper(services, ver):
    assert isinstance(services, list)
    assert len(services) == PANEL_ID_TO_K.get(ver)
    assert all([isinstance(service_id, int) for service_id in services])
