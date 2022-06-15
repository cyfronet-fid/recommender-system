# pylint: disable-all
import pytest


@pytest.mark.skip(
    reason="Tested in `test_ncf_pipeline` with less granularity. Not a priority for now."
)
def test_data_preparation_step(
    mongo,
    generate_users_and_services,
    ncf_pipeline_config,
):
    pass
