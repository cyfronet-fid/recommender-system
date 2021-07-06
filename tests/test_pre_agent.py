# pylint: disable-all

import pytest

from recommender.engine.agents.pre_agent.datasets.all_datasets import create_datasets
from recommender.engine.agents.pre_agent.models import NeuralColaborativeFilteringModel
from recommender.engine.agents.pre_agent.pre_agent import PreAgent
from recommender.engine.agents.pre_agent.training.common import pre_agent_training
from recommender.errors import InvalidRecommendationPanelIDError, MissingComponentError
from recommender.engine.agents.panel_id_to_services_number_mapping import PANEL_ID_TO_K

from recommender.engine.preprocessing import precalc_users_and_service_tensors
from recommender.models import User
from tests.factories.populate_database import populate_users_and_services


def test_pre_agent_call(mongo):
    # With no model case
    pre_agent = PreAgent()
    with pytest.raises(MissingComponentError):
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
    pre_agent = PreAgent()

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
