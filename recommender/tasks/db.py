"""Example celery tasks"""

from recommender.extensions import celery
from recommender.services.mp_dump import drop_mp_dump, load_mp_dump


@celery.task
def handle_db_dump(data):
    """Drops the MP part of the mongoDB and loads a new dump"""
    drop_mp_dump()
    load_mp_dump(data)
