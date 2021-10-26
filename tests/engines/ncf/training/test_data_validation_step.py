# pylint: disable-all
import pytest


@pytest.mark.skip(
    reason="Tested in `test_ncf_pipeline` with less granularity. Not a priority for now."
)
def test_data_validation_step(
    mongo, generate_data, mock_autoencoders_pipeline_exec, ncf_pipeline_config
):
    pass
