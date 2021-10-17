# pylint: disable-all
import pytest

from recommender.engines.persistent_mixin import Persistent
from recommender.models.ml_component import MLComponent


# Below class can't be defined in the fixture because this class has to be
# pickable and local objects (like a class defined in the function) aren't
# pickable.
class MockClass(Persistent):
    def __init__(self, mock_field):
        self._mock_field = mock_field

    def mock_method(self):
        return self._mock_field


@pytest.fixture
def mocks():
    mock_object = (1, 2, 6, 7)
    mock_instance = MockClass(mock_object)
    mock_version = "mock_version"
    return mock_object, mock_instance, mock_version


def test_persistent_mixin(mongo, mocks):
    mock_object, mock_instance, mock_version = mocks

    assert len(MLComponent.objects) == 0
    mock_instance.save(version=mock_version)
    assert len(MLComponent.objects) == 1
    obj = MLComponent.objects.first()
    loaded_mock_instance = MockClass.load(version=mock_version)
    assert loaded_mock_instance.mock_method() == mock_object
