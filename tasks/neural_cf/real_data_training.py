# pylint: disable=redefined-outer-name, redefined-builtin, missing-function-docstring, no-member

"""Use this task to train Neural Colaborative Filtering model on data from
 development database"""

from mongoengine import connect

from recommender.engine.pre_agent.training.common import pre_agent_training
from settings import DevelopmentConfig


if __name__ == "__main__":
    connect(host=DevelopmentConfig.MONGODB_HOST)
    pre_agent_training()
