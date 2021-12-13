# pylint: disable=too-few-public-methods

"""Flask recommender factory"""

import os

from flask import Flask
import click
import sentry_sdk
from sentry_sdk.integrations.celery import CeleryIntegration
from sentry_sdk.integrations.flask import FlaskIntegration
from sentry_sdk.integrations.redis import RedisIntegration

from recommender.commands import migrate_command, train_command, db_command
from recommender.extensions import db, celery
from recommender.api import api
from recommender.models import User
from settings import config_by_name


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
    @app.cli.command("db")
    @click.argument("task", type=click.Choice(["seed", "drop_mp", "seed_faker"]))
    def tmp_db_command(task):
        db_command(task)

    @app.cli.command("train")
    @click.argument(
        "task", type=click.Choice(["ae", "ncf", "rl_v1", "rl_v2", "embedding", "all"])
    )
    def tmp_train_command(task):
        train_command(task)

    @app.cli.command("migrate")
    @click.argument(
        "task", type=click.Choice(["apply", "rollback", "list", "repopulate", "check"])
    )
    def tmp_migrate_command(task):
        migrate_command(task)


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
