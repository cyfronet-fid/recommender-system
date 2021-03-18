# pylint: disable-all
import pytest
from torch.utils.data import Subset

from recommender.engine.pre_agent.datasets import PreAgentDataset, load_last_dataset, TRAIN_DS_NAME, VALID_DS_NAME, TEST_DS_NAME
from recommender.engine.pre_agent.datasets.common import NoSavedDatasetError
from recommender.tasks.neural_networks import execute_pre_agent_preprocessing
from recommender.models import User, Service
from tests.factories.populate_database import populate_users_and_services


@pytest.mark.celery_app
@pytest.mark.celery_worker
def test_execute_pre_agent_preprocessing(mongo):
    populate_users_and_services(
        common_services_number=4,
        no_one_services_number=1,
        users_number=4,
        k_common_services_min=1,
        k_common_services_max=3,
    )

    for user in User.objects:
        assert user.tensor == []

    for service in Service.objects:
        assert service.tensor == []

    with pytest.raises(NoSavedDatasetError):
        load_last_dataset(TRAIN_DS_NAME)

    with pytest.raises(NoSavedDatasetError):
        load_last_dataset(VALID_DS_NAME)

    with pytest.raises(NoSavedDatasetError):
        load_last_dataset(TEST_DS_NAME)

    execute_pre_agent_preprocessing.delay()

    for user in User.objects:
        assert len(user.tensor) > 0

    for service in Service.objects:
        assert len(service.tensor) > 0

    assert isinstance(load_last_dataset(TRAIN_DS_NAME), Subset)
    assert isinstance(load_last_dataset(VALID_DS_NAME), Subset)
    assert isinstance(load_last_dataset(TEST_DS_NAME), Subset)






