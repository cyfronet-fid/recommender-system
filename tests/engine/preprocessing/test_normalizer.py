# pylint: disable-all
import pytest
import torch

from recommender.engines.autoencoders.ml_components.normalizer import (
    Normalizer,
    NormalizationMode,
)
from recommender.errors import (
    DifferentTypeObjectsInCollectionError,
    MissingDenseTensorError,
)
from tests.factories.marketplace import ServiceFactory, UserFactory


@pytest.fixture()
def batch():
    return torch.Tensor(
        [
            [0.7187, -0.2987, -0.7510],
            [1.2209, 0.1780, 2.7129],
            [-0.2852, -0.7077, 1.0437],
            [-0.8773, -0.8192, -2.2478],
        ]
    )


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


def test_normalizer_call_on_tensors_dim_wise(batch):
    normalizer = Normalizer(mode=NormalizationMode.DIMENSION_WISE)
    output, _ = normalizer(batch)
    desired = torch.Tensor(
        [
            [0.5886, -0.3645, -0.2768],
            [1.0000, 0.2172, 1.0000],
            [-0.2336, -0.8639, 0.3847],
            [-0.7186, -1.0000, -0.8286],
        ]
    )

    assert torch.allclose(output, desired, atol=10e-4)


def test_normalizer_call_on_tensors_norm_wise(batch):
    normalizer = Normalizer(mode=NormalizationMode.NORM_WISE)
    output, _ = normalizer(batch)
    print(output)
    desired = torch.Tensor(
        [
            [0.2411, -0.1002, -0.2520],
            [0.4097, 0.0597, 0.9103],
            [-0.0957, -0.2375, 0.3502],
            [-0.2944, -0.2749, -0.7542],
        ]
    )

    assert torch.allclose(output, desired, atol=10e-4)
