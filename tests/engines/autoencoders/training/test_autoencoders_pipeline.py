# pylint: disable-all

from recommender.engines.autoencoders.training.pipeline import AEPipeline
from tests.engines.autoencoders.fixtures import (
    generate_data,
    pipeline_config,
)


def test_autoencoders_pipeline(mongo, generate_data, pipeline_config):
    ae_pipeline = AEPipeline(pipeline_config)
    ae_pipeline()
