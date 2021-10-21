# pylint: disable=redefined-outer-name, redefined-builtin, missing-function-docstring, no-member
# pylint: disable=too-many-locals, invalid-name, too-many-statements

"""This is example of training and inferencing
 using Neural Colaborative Filtering model.
 """

from dotenv import dotenv_values, find_dotenv

from recommender.engine.agents.pre_agent.pre_agent import PRE_AGENT
from recommender.engine.agents.rl_agent.rl_agent import RL_AGENT
from recommender.engine.agents.rl_agent.training.common import rl_agent_training
from recommender.engine.agents.pre_agent.datasets.all_datasets import create_datasets
from recommender.engine.agents.pre_agent.training.common import pre_agent_training
from recommender.engines.autoencoders.training.data_preparation_step import (
    precalc_users_and_service_tensors,
)
from recommender.extensions import celery


@celery.task
def execute_training(_=None):
    agent_version = dotenv_values(find_dotenv()).get("AGENT_VERSION")
    trainings = {PRE_AGENT: pre_agent_training, RL_AGENT: rl_agent_training}
    training = trainings.get(agent_version, pre_agent_training)
    training()


@celery.task
def execute_pre_agent_preprocessing():
    precalc_users_and_service_tensors()
    create_datasets()
