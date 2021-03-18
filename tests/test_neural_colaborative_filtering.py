# pylint: disable-all
from unittest.mock import patch

from recommender.engine.pre_agent.datasets import create_datasets
from recommender.engine.pre_agent.preprocessing import precalc_users_and_service_tensors
from recommender.engine.pre_agent.training import pre_agent_training
from tests.factories.populate_database import populate_users_and_services


class TestNeuralCollaborativeFiltering:
    def test_neural_cf_training(self, mongo):
        populate_users_and_services(
            common_services_number=4,
            no_one_services_number=1,
            users_number=4,
            k_common_services_min=1,
            k_common_services_max=3,
        )

        precalc_users_and_service_tensors()
        create_datasets()

        with patch("matplotlib.pyplot.show"):
            pre_agent_training()
