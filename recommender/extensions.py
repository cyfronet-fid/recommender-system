"""Extensions for flask recommender"""

from flask_mongoengine import MongoEngine
from celery import Celery

db = MongoEngine()
celery = Celery()
