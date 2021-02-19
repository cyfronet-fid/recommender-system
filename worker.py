"""Celery factory"""

from recommender import init_celery

app = init_celery()
app.conf.imports = app.conf.imports + (
    "recommender.tasks.db",
    "recommender.tasks.neural_networks",
)
