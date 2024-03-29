# pylint: disable=too-few-public-methods, unused-argument, line-too-long, expression-not-assigned
# pylint: disable=unused-variable, abstract-class-instantiated

"""Flask recommender factory"""

import os

from flask import Flask
import click
import sentry_sdk
from sentry_sdk.integrations.celery import CeleryIntegration
from sentry_sdk.integrations.flask import FlaskIntegration
from sentry_sdk.integrations.redis import RedisIntegration
from celery.signals import setup_logging

from recommender.commands import (
    migrate_command,
    train_command,
    db_command,
    seed_command,
)
from recommender.commands.subscriber import subscribe
from recommender.extensions import db, celery
from recommender.api import api
from recommender.models import User
from recommender.services.metrics import calc_hitrate
from settings import config_by_name
from logger_config import apply_logging_config


def create_app():
    """Creates the flask recommender, initializes config in
    the proper ENV and initializes flask-restx"""
    env = os.environ["FLASK_ENV"]
    init_sentry_flask()

    app = Flask(__name__)
    app.config.from_object(config_by_name[env])

    if not env == "testing":
        apply_logging_config()

    _register_extensions(app)
    api.init_app(app)
    init_celery(app)
    _register_commands(app, env)

    return app


def _register_commands(app, env: str):
    @app.cli.command("hitrate")
    @click.argument("engine_version", required=False)
    @click.argument("panel_id", required=False)
    def tmp_hitrate_command(engine_version=None, panel_id=None):
        calc_hitrate(engine_version, panel_id)

    @app.cli.command("db", help="Database related tasks (without seeding).")
    @click.argument(
        "task",
        type=click.Choice(["drop_mp", "drop_models", "regenerate_sarses"]),
    )
    def tmp_db_command(task):
        db_command(task)

    if env != "production":

        @app.cli.command("seed", help="Tasks for seeding database.")
        @click.argument(
            "task",
            type=click.Choice(["seed", "seed_faker"]),
        )
        def tmp_seed_command(task):
            seed_command(task)

    @app.cli.command("train", help="Run training routine.")
    @click.argument("task", type=click.Choice(["ae", "embedding", "ncf", "rl", "all"]))
    def tmp_train_command(task):
        agreed = True
        if task == "all":
            agreed = click.confirm(
                "Calling this command will delete all of the existing ML models, do you agree?"
            )

        train_command(task) if agreed else print("Aborting...")

    # pylint: disable=too-many-arguments
    @app.cli.command("migrate", help="Migrate database.")
    @click.argument(
        "task", type=click.Choice(["apply", "rollback", "list", "repopulate", "check"])
    )
    def tmp_migrate_command(task):
        migrate_command(task)

    @app.cli.command(
        "subscribe", help="Subscribe to databus for events (like user actions)."
    )
    @click.option(
        "--host",
        envvar="RS_DATABUS_HOST",
        default="127.0.0.1",
        show_default=True,
        show_envvar=True,
    )
    @click.option(
        "--port",
        default=61613,
        envvar="RS_DATABUS_PORT",
        show_default=True,
        show_envvar=True,
    )
    @click.option(
        "--username",
        default="admin",
        envvar="RS_DATABUS_USERNAME",
        show_default=True,
        show_envvar=True,
    )
    @click.option(
        "--password",
        default="admin",
        envvar="RS_DATABUS_PASSWORD",
        show_default=True,
        show_envvar=True,
    )
    @click.option(
        "--topic",
        default="/topic/user_actions",
        envvar="RS_DATABUS_SUBSCRIPTION_TOPIC",
        show_default=True,
        show_envvar=True,
    )
    @click.option(
        "--subscription-id",
        envvar="RS_DATABUS_SUBSCRIPTION_ID",
        help="Subscription id should be unique. "
        "If not specified random string will be generated",
        show_envvar=True,
    )
    @click.option(
        "--ssl/--no-ssl",
        default=True,
        envvar="RS_DATABUS_SSL",
        show_envvar=True,
    )
    def tmp_subscribe(host, port, username, password, topic, subscription_id, ssl):
        subscribe(host, port, username, password, topic, subscription_id, ssl)


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
