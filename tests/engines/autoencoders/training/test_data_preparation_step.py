# pylint: disable-all
import pytest


@pytest.mark.skip(
    reason="Tested in `test_autoencoders_pipeline` with less granularity. Not a priority for now."
)
def test_data_preparation_step(
    mongo, generate_data, mock_autoencoders_pipeline_exec, pipeline_config
):
    pass
