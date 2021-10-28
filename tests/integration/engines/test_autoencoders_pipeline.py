# pylint: disable-all

from recommender.engines.autoencoders.training.pipeline import AEPipeline


def test_autoencoders_pipeline(mongo, generate_users_and_services, ae_pipeline_config):
    ae_pipeline = AEPipeline(ae_pipeline_config)
    ae_pipeline()
