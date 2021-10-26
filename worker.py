"""Celery factory"""

from recommender import init_celery, init_sentry_celery

init_sentry_celery()
app = init_celery()

custom_tasks = ("recommender.tasks",)

app.conf.imports += custom_tasks
