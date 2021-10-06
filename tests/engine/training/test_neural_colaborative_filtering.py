# pylint: disable-all
from recommender.engine.agents.pre_agent.datasets.all_datasets import create_datasets
from recommender.engine.agents.pre_agent.training.common import pre_agent_training
from recommender.engine.preprocessing import precalc_users_and_service_tensors
from tests.factories.populate_database import populate_users_and_services


class TestNeuralCollaborativeFiltering:
    def test_neural_cf_training(self, mongo):
        populate_users_and_services(
            common_services_number=9,
            no_one_services_number=9,
            users_number=5,
            k_common_services_min=5,
            k_common_services_max=7,
        )

        precalc_users_and_service_tensors()
        create_datasets()

        pre_agent_training()
