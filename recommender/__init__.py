# pylint: disable=too-few-public-methods

"""Flask recommender factory"""

import os

from flask import Flask
import click
import sentry_sdk
from sentry_sdk.integrations.celery import CeleryIntegration
from sentry_sdk.integrations.flask import FlaskIntegration
from sentry_sdk.integrations.redis import RedisIntegration

from recommender.extensions import db, celery
from recommender.api import api
from recommender.models import User
from settings import config_by_name

from .commands import seed_faker, seed_db, execute_training, drop_mp_dump_task


def create_app():
    """Creates the flask recommender, initializes config in
    the proper ENV and initializes flask-restx"""

    init_sentry_flask()

    app = Flask(__name__)
    app.config.from_object(config_by_name[os.environ["FLASK_ENV"]])

    _register_extensions(app)
    api.init_app(app)
    init_celery(app)

    _register_commands(app)

    return app


def _register_commands(app):
    @app.cli.command("seed_faker")
    def seed_faker_command():
        seed_faker()

    @app.cli.command("seed_mp_dump")
    def seed_mp_dump_command():
        seed_db()

    @app.cli.command("drop_mp_dump")
    def drop_mp_dump_command():
        drop_mp_dump_task()

    @app.cli.command("execute_training")
    @click.argument("task")
    def execute_training_command(task):
        execute_training(task)


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


def init_sentry_flask():
    """Initializes sentry-flask integration"""

    sentry_sdk.init(
        integrations=[FlaskIntegration()],
        traces_sample_rate=1.0,
    )


def init_sentry_celery():
    """Initializes sentry-celery integration"""

    sentry_sdk.init(
        integrations=[CeleryIntegration(), RedisIntegration()],
        traces_sample_rate=1.0,
    )
