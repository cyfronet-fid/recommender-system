# pylint: disable-all

import pytest

from recommender.engine.pre_agent.models import NeuralColaborativeFilteringModel
from recommender.engine.pre_agent.models.common import NoSavedModuleError, load_last_module
from recommender.engine.pre_agent.models import NEURAL_CF
from recommender.engine.pre_agent.datasets import create_datasets
from recommender.engine.pre_agent.preprocessing import precalc_users_and_service_tensors
from recommender.tasks.neural_networks import execute_pre_agent_training
from tests.factories.populate_database import populate_users_and_services


@pytest.mark.celery_app
@pytest.mark.celery_worker
def test_execute_pre_agent_training(mongo):
    populate_users_and_services(
        common_services_number=4,
        no_one_services_number=1,
        users_number=4,
        k_common_services_min=1,
        k_common_services_max=3,
    )

    precalc_users_and_service_tensors()
    create_datasets()

    with pytest.raises(NoSavedModuleError):
        load_last_module(NEURAL_CF)

    execute_pre_agent_training.delay()

    assert isinstance(load_last_module(NEURAL_CF), NeuralColaborativeFilteringModel)
