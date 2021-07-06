# pylint: disable=redefined-outer-name, redefined-builtin, missing-function-docstring, no-member
# pylint: disable=too-many-locals, invalid-name, too-many-statements

"""This is example of training and inferencing
 using Neural Colaborative Filtering model.
 """
from flask import current_app

from recommender.engine.agents.pre_agent.pre_agent import PRE_AGENT, PreAgent
from recommender.engine.agents.rl_agent.rl_agent import RL_AGENT, RLAgent
from recommender.engine.agents.rl_agent.training.common import rl_agent_training
from recommender.engine.agents.pre_agent.datasets.all_datasets import create_datasets
from recommender.engine.agents.pre_agent.training.common import pre_agent_training
from recommender.engine.preprocessing import precalc_users_and_service_tensors
from recommender.extensions import celery


@celery.task
def execute_pre_agent_training():
    pre_agent_training()


@celery.task
def execute_rl_agent_training():
    rl_agent_training()


@celery.task
def execute_training(_=None):
    if current_app.config["AGENT_VERSION"] == PRE_AGENT:
        pre_agent_training()
    elif current_app.config["AGENT_VERSION"] == RL_AGENT:
        rl_agent_training()
    else:
        current_app.recommender_engine = PreAgent()


@celery.task
def reload(_=None):
    with current_app.app_context():
        if current_app.config["AGENT_VERSION"] == PRE_AGENT:
            current_app.recommender_engine = PreAgent()
        elif current_app.config["AGENT_VERSION"] == RL_AGENT:
            current_app.recommender_engine = RLAgent()
        else:
            current_app.recommender_engine = PreAgent()


@celery.task
def execute_pre_agent_preprocessing():
    precalc_users_and_service_tensors()
    create_datasets()
