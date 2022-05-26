# pylint: disable-all

import random
import pytest
from recommender.engines.random.inference.random_inference_component import (
    RandomInferenceComponent,
)
from recommender.engines.panel_id_to_services_number_mapping import PANEL_ID_TO_K
from recommender.models import Service, User
from recommender.errors import (
    InsufficientRecommendationSpaceError,
    InvalidRecommendationPanelIDError,
)


@pytest.mark.parametrize("ver", ["v1", "v2"])
def test_random_inference_component(mongo, generate_users_and_services, ver):
    K = PANEL_ID_TO_K[ver]
    inference_component = RandomInferenceComponent(K=K)
    assert isinstance(inference_component, RandomInferenceComponent)
    assert isinstance(inference_component.engine_name, str)

    elastic_services = [service.id for service in Service.objects]
    user = random.choice(list(User.objects))

    contexts = [
        {
            "panel_id": ver,
            "elastic_services": elastic_services,
            "search_data": {},
            "user_id": user.id,
        },
        {
            "panel_id": ver,
            "elastic_services": elastic_services,
            "search_data": {},
            "user_id": -1,
        },
        {
            "panel_id": ver,
            "elastic_services": elastic_services,
            "search_data": {},
        },
        {
            "panel_id": ver,
            "elastic_services": elastic_services,
        },
        {
            "elastic_services": elastic_services,
        },
    ]

    for context in contexts:
        services_ids_1 = inference_component(context)
        _check_proper(services_ids_1, elastic_services, ver)

        services_ids_2 = inference_component(context)
        _check_proper(services_ids_2, elastic_services, ver)
        assert services_ids_1 != services_ids_2

    # 2. case InsufficientRecommendationSpaceError - in all cases
    contexts = [
        {"elastic_services": None},
        {"elastic_services": []},
        {"elastic_services": ()},
        {"elastic_services": [random.choice(elastic_services)]},
    ]

    for context in contexts:
        with pytest.raises(InsufficientRecommendationSpaceError):
            inference_component(context)

    # 3. case InsufficientRecommendationSpaceError - depends on the version
    contexts = [
        {"elastic_services": random.choices(elastic_services, k=2)},
        {"elastic_services": random.choices(elastic_services, k=3)},
    ]

    for context in contexts:
        if ver == "v1" and len(context["elastic_services"]) == 2:
            with pytest.raises(InsufficientRecommendationSpaceError):
                inference_component(context)
        else:
            services_ids_1 = inference_component(context)
            _check_proper(services_ids_1, elastic_services, ver)


def _check_proper(services, elastic_services, ver):
    assert isinstance(services, list)
    assert len(services) == PANEL_ID_TO_K.get(ver)
    assert all([isinstance(service_id, int) for service_id in services])
    assert all(service in elastic_services for service in services)


@pytest.mark.parametrize("K", [0, 1, 4, 5])
def test_incorrect_init_of_anonymous_inference_component(mongo, K):
    with pytest.raises(InvalidRecommendationPanelIDError):
        RandomInferenceComponent(K=K)
