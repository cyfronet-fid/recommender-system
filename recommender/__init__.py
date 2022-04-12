# pylint: disable=too-few-public-methods, unused-argument

"""Flask recommender factory"""

import os

from flask import Flask
import click
import sentry_sdk
from sentry_sdk.integrations.celery import CeleryIntegration
from sentry_sdk.integrations.flask import FlaskIntegration
from sentry_sdk.integrations.redis import RedisIntegration
from celery.signals import setup_logging

from recommender.commands import migrate_command, train_command, db_command
from recommender.extensions import db, celery
from recommender.api import api
from recommender.models import User
from recommender.services.metrics import calc_hitrate
from settings import config_by_name
from logger_config import apply_logging_config


def create_app():
    """Creates the flask recommender, initializes config in
    the proper ENV and initializes flask-restx"""

    init_sentry_flask()

    app = Flask(__name__)
    app.config.from_object(config_by_name[os.environ["FLASK_ENV"]])
    if not os.environ["FLASK_ENV"] == "testing":
        apply_logging_config()

    _register_extensions(app)
    api.init_app(app)
    init_celery(app)
    _register_commands(app)

    return app


def _register_commands(app):
    @app.cli.command("hitrate")
    @click.argument("engine_version", required=False)
    @click.argument("panel_id", required=False)
    def tmp_hitrate_command(engine_version=None, panel_id=None):
        calc_hitrate(engine_version, panel_id)

    @app.cli.command("db")
    @click.argument(
        "task",
        type=click.Choice(
            ["seed", "drop_mp", "drop_models", "seed_faker", "regenerate_sarses"]
        ),
    )
    def tmp_db_command(task):
        db_command(task)

    @app.cli.command("train")
    @click.argument("task", type=click.Choice(["ae", "embedding", "ncf", "rl", "all"]))
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


@setup_logging.connect()
def configurate_celery_task_logger(**kwargs):
    """Celery won’t configure the loggers if this signal is connected,
    allowing the logger to utilize our configuration"""
