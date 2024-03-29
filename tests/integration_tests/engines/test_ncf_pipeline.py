# pylint: disable-all

from recommender.engines.ncf.training.pipeline import NCFPipeline


def test_ncf_pipeline(
    mongo, generate_users_and_services, ncf_pipeline_config, embedding_exec
):
    ncf_pipeline = NCFPipeline(ncf_pipeline_config)
    ncf_pipeline()
