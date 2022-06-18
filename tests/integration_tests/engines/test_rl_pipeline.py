# pylint: disable-all
import pytest

from recommender.engines.rl.training.pipeline import RLPipeline


def test_rl_pipeline(
    mongo,
    generate_users_and_services,
    rl_pipeline_v1_config,
    rl_pipeline_v2_config,
):
    RLPipeline(rl_pipeline_v1_config)()
    RLPipeline(rl_pipeline_v2_config)()
