# pylint: disable-all

import pytest

from recommender.engine.utils import NoSavedDatasetError, load_last_dataset


def test_load_last_dataset(mongo):
    with pytest.raises(NoSavedDatasetError):
        load_last_dataset("placeholder_name")
