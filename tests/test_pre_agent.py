# pylint: disable-all
import sys

import pytest

from recommender.engine.panel_id_to_services_number_mapping import PANEL_ID_TO_K
from recommender.engine.pre_agent.datasets import create_datasets
from recommender.engine.pre_agent.models import NeuralColaborativeFilteringModel
from recommender.engine.pre_agent.pre_agent import _get_not_accessed_services, _services_to_ids, \
    _fill_candidate_services, \
    PreAgentRecommender, UntrainedPreAgentError, InvalidRecommendationPanelIDError
from recommender.engine.pre_agent.preprocessing import precalc_users_and_service_tensors
from recommender.engine.pre_agent.training import pre_agent_training
from recommender.models import User
from recommender.models import Service
from tests.factories.marketplace import UserFactory, ServiceFactory
from tests.factories.populate_database import populate_users_and_services


def test_get_not_accessed_services(mongo):
    user = UserFactory()
    print(f"user.accessed_services: {[s.id for s in user.accessed_services]}", file=sys.stderr)
    not_accessed_services = ServiceFactory.create_batch(10)
    not_accessed_services_set = set(list(not_accessed_services))

    print(f"not_accessed_services_set: {[s.id for s in not_accessed_services_set]}", file=sys.stderr)

    output_set = set(list(_get_not_accessed_services(user)))
    print(f"output_set: {[s.id for s in output_set]}", file=sys.stderr)

    assert not_accessed_services_set == output_set


def test_services_to_ids(mongo):
    services = ServiceFactory.create_batch(3)
    services_ids = [services[0].id, services[1].id, services[2].id]

    output = _services_to_ids(services)

    assert services_ids == output


def test_fill_candidate_services(mongo):
    all_services = list(ServiceFactory.create_batch(5))

    for required_services_no in range(1, 4):
        for candidate_services_no in range(1, required_services_no+1):
            candidate_services = all_services[:candidate_services_no]
            filled_services = _fill_candidate_services(
                candidate_services, required_services_no
            )

            assert isinstance(filled_services, list)
            assert len(filled_services) == required_services_no
            assert all([isinstance(s, Service) for s in filled_services])


def test_pre_agent_call(mongo):
    # With no model case
    pre_agent = PreAgentRecommender()
    with pytest.raises(UntrainedPreAgentError):
        place_holder_context = {"placeholder_key": "placeholder_value"}
        pre_agent.call(place_holder_context)

    # Create data and model
    populate_users_and_services(
        common_services_number=4,
        no_one_services_number=5,
        users_number=4,
        k_common_services_min=1,
        k_common_services_max=3,
    )

    precalc_users_and_service_tensors()
    create_datasets()
    pre_agent_training()

    # With model case
    pre_agent = PreAgentRecommender()

    for panel_id_version in list(PANEL_ID_TO_K.keys()):
        user = User.objects.first()
        context = {
            "panel_id": panel_id_version,
            "search_data": {},
            "logged_user": True,
            "user_id": user.id
        }

        services_ids_1 = pre_agent.call(context)
        assert isinstance(pre_agent.neural_cf_model, NeuralColaborativeFilteringModel)

        assert isinstance(services_ids_1, list)
        assert len(services_ids_1) == PANEL_ID_TO_K.get(context["panel_id"])
        assert all([isinstance(service_id, int) for service_id in services_ids_1])

        services_ids_2 = pre_agent.call(context)
        assert services_ids_1 == services_ids_2

    for panel_id_version in list(PANEL_ID_TO_K.keys()):
        context = {
            "panel_id": panel_id_version,
            "search_data": {},
            "logged_user": True,
            "user_id": -1
        }

        services_ids_1 = pre_agent.call(context)
        assert isinstance(pre_agent.neural_cf_model, NeuralColaborativeFilteringModel)

        assert isinstance(services_ids_1, list)
        assert len(services_ids_1) == PANEL_ID_TO_K.get(context["panel_id"])
        assert all([isinstance(service_id, int) for service_id in services_ids_1])

        services_ids_2 = pre_agent.call(context)
        assert services_ids_1 != services_ids_2

    for panel_id_version in list(PANEL_ID_TO_K.keys()):
        context = {
            "panel_id": panel_id_version,
            "search_data": {},
            "logged_user": False
        }

        services_ids_1 = pre_agent.call(context)
        assert isinstance(pre_agent.neural_cf_model, NeuralColaborativeFilteringModel)

        assert isinstance(services_ids_1, list)
        assert len(services_ids_1) == PANEL_ID_TO_K.get(context["panel_id"])
        assert all([isinstance(service_id, int) for service_id in services_ids_1])

        services_ids_2 = pre_agent.call(context)
        assert services_ids_1 != services_ids_2

    context = {
        "panel_id": "invalid_panel_id",
        "search_data": {},
        "logged_user": False
    }

    with pytest.raises(InvalidRecommendationPanelIDError):
        pre_agent.call(context)
