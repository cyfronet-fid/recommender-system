"""Example celery tasks"""

from recommender.extensions import celery


@celery.task
def dummy_task():
    """Example task to show how to configure celery tasks"""
    return "OK"
