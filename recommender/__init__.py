# pylint: disable=too-few-public-methods

"""Flask recommender factory"""

import os
from flask import Flask

from recommender.engine.pre_agent.models.common import load_last_module
from recommender.engine.pre_agent.models.neural_colaborative_filtering import NEURAL_CF
from recommender.engine.pre_agent.pre_agent import PreAgentRecommender
from recommender.extensions import db, celery
from recommender.api import api
from settings import config_by_name


def create_app():
    """Creates the flask recommender, initializes config in
    the proper ENV and initializes flask-restx"""

    app = Flask(__name__)
    app.config.from_object(config_by_name[os.environ["FLASK_ENV"]])

    _register_extensions(app)
    api.init_app(app)
    init_celery(app)

    init_recommender_engine(app)

    return app


def _register_extensions(app):
    db.init_app(app)


def init_celery(app=None):
    """Initializes celery"""

    app = app or create_app()

    if os.environ["FLASK_ENV"] == "testing":
        celery.conf.update(task_always_eager=True)
    else:
        celery.conf.update(
            broker_url=app.config["REDIS_HOST"],
            result_backend=app.config["REDIS_HOST"],
        )

    class ContextTask(celery.Task):
        """Make celery tasks work with Flask recommender context"""

        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery


def init_recommender_engine(app):
    """Instantiate a recommender engine in the Flask app"""
    app.recommender_engine = PreAgentRecommender()
