# pylint: disable-all

import pytest

from recommender.engine.agents.pre_agent.datasets.all_datasets import create_datasets
from recommender.engine.agents.pre_agent.models import (
    NEURAL_CF,
    NeuralColaborativeFilteringModel,
)
from recommender.engine.utils import NoSavedModuleError, load_last_module
from recommender.engine.preprocessing import precalc_users_and_service_tensors
from recommender.tasks.neural_networks import execute_pre_agent_training
from tests.factories.populate_database import populate_users_and_services


@pytest.mark.celery_app
@pytest.mark.celery_worker
def test_execute_pre_agent_training(mongo):
    populate_users_and_services(
        common_services_number=9,
        no_one_services_number=9,
        users_number=5,
        k_common_services_min=5,
        k_common_services_max=7,
    )

    precalc_users_and_service_tensors()
    create_datasets()

    with pytest.raises(NoSavedModuleError):
        load_last_module(NEURAL_CF)

    task = execute_pre_agent_training.delay()
    task.wait(timeout=None, interval=0.5)

    assert isinstance(load_last_module(NEURAL_CF), NeuralColaborativeFilteringModel)
