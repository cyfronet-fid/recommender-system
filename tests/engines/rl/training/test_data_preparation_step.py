# pylint: disable-all
import pytest


@pytest.mark.skip(reason="Not a priority for now.")
def test_data_preparation_step(
    mongo,
    generate_users_and_services,
):
    pass
