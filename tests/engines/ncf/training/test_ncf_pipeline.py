# pylint: disable-all

from recommender.engines.ncf.training.pipeline import NCFPipeline

# TODO: you don't have to import fixtures with pytest, it's done in some other way
from ..fixtures import generate_data, pipeline_config, mock_autoencoders_pipeline_exec


def test_ncf_pipeline(
    mongo, generate_data, pipeline_config, mock_autoencoders_pipeline_exec
):
    ncf_pipeline = NCFPipeline(pipeline_config)
    ncf_pipeline()
