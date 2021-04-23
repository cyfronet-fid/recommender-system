# pylint: disable-all
import sys

import pytest

from recommender.engine.pre_agent.datasets.all_datasets import create_datasets
from recommender.engine.pre_agent.training.common import pre_agent_training
from recommender.engine.panel_id_to_services_number_mapping import PANEL_ID_TO_K
from recommender.engine.pre_agent.models import NeuralColaborativeFilteringModel
from recommender.engine.pre_agent.pre_agent import (
    _services_to_ids,
    _fill_candidate_services,
    PreAgentRecommender,
    UntrainedPreAgentError,
    InvalidRecommendationPanelIDError,
)
from recommender.engine.pre_agent.preprocessing import precalc_users_and_service_tensors
from recommender.models import User
from recommender.models import Service
from tests.factories.marketplace import UserFactory, ServiceFactory
from tests.factories.populate_database import populate_users_and_services


def test_services_to_ids(mongo):
    services = ServiceFactory.create_batch(3)
    services_ids = [services[0].id, services[1].id, services[2].id]

    output = _services_to_ids(services)

    assert services_ids == output


def test_fill_candidate_services(mongo):
    all_services = list(ServiceFactory.create_batch(5))

    for required_services_no in range(1, 4):
        for candidate_services_no in range(1, required_services_no + 1):
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
        common_services_number=9,
        no_one_services_number=9,
        users_number=5,
        k_common_services_min=5,
        k_common_services_max=7,
    )

    precalc_users_and_service_tensors()
    create_datasets()
    pre_agent_training()

    # With model case
    pre_agent = PreAgentRecommender()

    for panel_id_version in list(PANEL_ID_TO_K.keys()):
        user = User.objects.first()
        context = {"panel_id": panel_id_version, "search_data": {}, "user_id": user.id}

        services_ids_1 = pre_agent.call(context)
        assert isinstance(pre_agent.neural_cf_model, NeuralColaborativeFilteringModel)

        assert isinstance(services_ids_1, list)
        assert len(services_ids_1) == PANEL_ID_TO_K.get(context["panel_id"])
        assert all([isinstance(service_id, int) for service_id in services_ids_1])

        services_ids_2 = pre_agent.call(context)
        assert services_ids_1 == services_ids_2

    for panel_id_version in list(PANEL_ID_TO_K.keys()):
        context = {"panel_id": panel_id_version, "search_data": {}, "user_id": -1}

        services_ids_1 = pre_agent.call(context)
        assert isinstance(pre_agent.neural_cf_model, NeuralColaborativeFilteringModel)

        assert isinstance(services_ids_1, list)
        assert len(services_ids_1) == PANEL_ID_TO_K.get(context["panel_id"])
        assert all([isinstance(service_id, int) for service_id in services_ids_1])

        services_ids_2 = pre_agent.call(context)
        assert services_ids_1 != services_ids_2

    for panel_id_version in list(PANEL_ID_TO_K.keys()):
        context = {"panel_id": panel_id_version, "search_data": {}}

        services_ids_1 = pre_agent.call(context)
        assert isinstance(pre_agent.neural_cf_model, NeuralColaborativeFilteringModel)

        assert isinstance(services_ids_1, list)
        assert len(services_ids_1) == PANEL_ID_TO_K.get(context["panel_id"])
        assert all([isinstance(service_id, int) for service_id in services_ids_1])

        services_ids_2 = pre_agent.call(context)
        assert services_ids_1 != services_ids_2

    context = {"panel_id": "invalid_panel_id", "search_data": {}}

    with pytest.raises(InvalidRecommendationPanelIDError):
        pre_agent.call(context)
