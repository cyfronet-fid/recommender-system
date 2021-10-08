# pylint: disable-all
import pytest
import torch

from recommender.engine.preprocessing.normalizer import Normalizer
from recommender.errors import (
    DifferentTypeObjectsInCollectionError,
    MissingDenseTensorError,
)
from tests.factories.marketplace import ServiceFactory, UserFactory


def test_normalizer(mongo):
    DENSE_DIM = 64

    services = ServiceFactory.create_batch(10)

    normalizer = Normalizer()

    with pytest.raises(MissingDenseTensorError):
        normalizer(services, save_cache=False)

    for service in services:
        service.dense_tensor = (torch.rand((DENSE_DIM,)) * 10).tolist()
        service.save()

    dense_matrix_from_cache = torch.Tensor(
        [service.dense_tensor for service in services]
    )

    normalized_matrix, _ = normalizer(services, save_cache=False)

    assert normalized_matrix.shape == dense_matrix_from_cache.shape
    assert not torch.all(torch.isclose(normalized_matrix, dense_matrix_from_cache))
    assert torch.all(normalized_matrix <= 1).item()

    normalized_matrix, _ = normalizer(services, save_cache=True)

    dense_matrix_from_cache = torch.Tensor(
        [service.dense_tensor for service in services]
    )

    assert normalized_matrix.shape == dense_matrix_from_cache.shape
    assert torch.all(torch.isclose(normalized_matrix, dense_matrix_from_cache))
    assert torch.all(normalized_matrix <= 1).item()

    users = UserFactory.create_batch(10)
    different_objects = list(users) + list(services)
    with pytest.raises(DifferentTypeObjectsInCollectionError):
        normalizer(different_objects, save_cache=False)
