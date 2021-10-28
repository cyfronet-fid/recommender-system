# pylint: disable-all

from recommender.engines.rl.training.pipeline import RLPipeline


def test_rl_pipeline(
    mongo, generate_users_and_services, rl_pipeline_config, embedding_exec
):
    rl_pipeline = RLPipeline(rl_pipeline_config)
    rl_pipeline()
