# pylint: disable-all
import pytest

from recommender.engines.ncf.training.pipeline import NCFPipeline


@pytest.mark.skip(reason="TODO")
def test_ncf_pipeline(mongo, generate_users_and_services, ncf_pipeline_config):
    ncf_pipeline = NCFPipeline(ncf_pipeline_config)
    ncf_pipeline()
