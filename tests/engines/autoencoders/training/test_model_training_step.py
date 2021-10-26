# pylint: disable-all

import pytest


@pytest.mark.skip(
    reason="Tested in `test_autoencoders_pipeline` with less granularity. Not a priority for now."
)
def test_model_training_step(mongo, generate_data, ae_pipeline_config):
    pass
