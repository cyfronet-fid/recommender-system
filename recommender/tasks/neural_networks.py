# pylint: disable=redefined-outer-name, redefined-builtin, missing-function-docstring, no-member
# pylint: disable=too-many-locals, invalid-name, too-many-statements

"""This is example of training and inferencing
 using Neural Colaborative Filtering model.
 """

from recommender.engine.pre_agent.datasets import create_datasets
from recommender.engine.pre_agent.preprocessing import precalc_users_and_service_tensors
from recommender.engine.pre_agent.training import pre_agent_training
from recommender.extensions import celery


@celery.task
def execute_pre_agent_training():
    pre_agent_training()


@celery.task
def execute_pre_agent_preprocessing():
    precalc_users_and_service_tensors()
    create_datasets()
