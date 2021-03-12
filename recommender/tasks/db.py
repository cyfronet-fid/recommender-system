"""Example celery tasks"""
from recommender.engine.pre_agent.datasets import create_datasets
from recommender.engine.pre_agent.preprocessing import precalc_users_and_service_tensors
from recommender.extensions import celery
from recommender.services.mp_dump import drop_mp_dump, load_mp_dump


@celery.task
def handle_db_dump(data):
    """Drops the MP part of the mongoDB and loads a new dump"""
    drop_mp_dump()
    load_mp_dump(data)
    precalc_users_and_service_tensors()
    create_datasets()
