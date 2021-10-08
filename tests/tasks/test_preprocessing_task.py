# pylint: disable-all
import pytest
from torch.utils.data import Subset

from recommender.engine.datasets.autoencoders import (
    get_autoencoder_dataset_name,
)
from recommender.engine.preprocessing import SERVICES
from recommender.engine.utils import (
    NoSavedDatasetError,
    load_last_dataset,
    TRAIN,
    VALID,
    TEST,
)
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
        assert user.one_hot_tensor == []

    for service in Service.objects:
        assert service.one_hot_tensor == []

    with pytest.raises(NoSavedDatasetError):
        load_last_dataset(get_autoencoder_dataset_name(SERVICES, TRAIN))

    with pytest.raises(NoSavedDatasetError):
        load_last_dataset(get_autoencoder_dataset_name(SERVICES, VALID))

    with pytest.raises(NoSavedDatasetError):
        load_last_dataset(get_autoencoder_dataset_name(SERVICES, TEST))

    execute_pre_agent_preprocessing.delay()

    for user in User.objects:
        assert len(user.one_hot_tensor) > 0

    for service in Service.objects:
        assert len(service.one_hot_tensor) > 0

    assert isinstance(
        load_last_dataset(get_autoencoder_dataset_name(SERVICES, TRAIN)), Subset
    )
    assert isinstance(
        load_last_dataset(get_autoencoder_dataset_name(SERVICES, VALID)), Subset
    )
    assert isinstance(
        load_last_dataset(get_autoencoder_dataset_name(SERVICES, TEST)), Subset
    )
