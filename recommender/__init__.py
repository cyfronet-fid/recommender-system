# pylint: disable=too-few-public-methods

"""Flask recommender factory"""

import os
from flask import Flask

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
    return app


def _register_extensions(app):
    db.init_app(app)


def init_celery(app=None):
    """Initializes celery"""

    app = app or create_app()
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
