# pylint: disable-all
import pandas as pd
import pytest
import torch

from recommender.engine.models.autoencoders import AutoEncoder
from recommender.engines.autoencoders.ml_components.embedder import Embedder
from recommender.errors import MissingDenseTensorError, MissingOneHotTensorError
from tests.factories.marketplace import ServiceFactory


@pytest.fixture
def services(mongo):
    return [ServiceFactory(id=i) for i in range(10)]


def test_embedder(mongo, services):
    ONE_HOT_DIM = 256
    DENSE_DIM = 64

    autoencoder_mock = AutoEncoder(ONE_HOT_DIM, DENSE_DIM)
    embedder = Embedder(autoencoder_mock)

    assert embedder.one_hot_dim == ONE_HOT_DIM
    assert embedder.dense_dim == DENSE_DIM
    assert all(p.requires_grad == False for p in embedder.network.parameters())

    with pytest.raises(MissingOneHotTensorError):
        embedder(services, use_cache=False, save_cache=False)

    for service in services:
        service.one_hot_tensor = torch.randint(0, 2, (ONE_HOT_DIM,)).tolist()
        service.save()

    one_hot_matrix = torch.Tensor([service.one_hot_tensor for service in services])
    valid_dense_matrix = autoencoder_mock.encoder(one_hot_matrix)

    with pytest.raises(MissingDenseTensorError):
        embedder(services, use_cache=True, save_cache=False)

    dense_matrix, index_id_map = embedder(services, use_cache=False, save_cache=False)

    assert isinstance(index_id_map, pd.DataFrame)
    assert all(index_id_map.index.values == index_id_map.values.reshape(-1))

    assert valid_dense_matrix.shape == dense_matrix.shape
    assert torch.all(torch.isclose(valid_dense_matrix, dense_matrix)).item()

    assert all(service.dense_tensor == [] for service in services)

    dense_matrix, index_id_map = embedder(services, use_cache=False, save_cache=True)

    assert valid_dense_matrix.shape == dense_matrix.shape
    assert torch.all(torch.isclose(valid_dense_matrix, dense_matrix)).item()

    dense_matrix_from_cache = torch.Tensor(
        [service.dense_tensor for service in services]
    )

    assert valid_dense_matrix.shape == dense_matrix_from_cache.shape
    assert torch.all(torch.isclose(valid_dense_matrix, dense_matrix_from_cache)).item()

    for service in services:
        service.one_hot_tensor = torch.randint(0, 2, (ONE_HOT_DIM,)).tolist()
        service.save()

    dense_matrix_use_cache, index_id_map = embedder(
        services, use_cache=True, save_cache=False
    )

    assert dense_matrix_from_cache.shape == dense_matrix_use_cache.shape
    assert torch.all(
        torch.isclose(dense_matrix_from_cache, dense_matrix_use_cache)
    ).item()

    for service in services:
        service.dense_tensor = []
        service.save()

    with pytest.raises(MissingDenseTensorError):
        embedder(services, use_cache=True, save_cache=False)

    dense_matrix = embedder.network(one_hot_matrix)
    assert valid_dense_matrix.shape == dense_matrix.shape
    assert torch.all(torch.isclose(valid_dense_matrix, dense_matrix)).item()
