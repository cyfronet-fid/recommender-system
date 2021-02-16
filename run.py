# pylint: disable=missing-class-docstring, too-few-public-methods

"""Application launch module"""

import os
from flask import Flask
from celery import Celery
from app.api.v1 import use_api

app = Flask(__name__)


def make_celery(flask_app):
    """celery initialization function"""
    celery_instance = Celery(
        flask_app.import_name,
        backend=flask_app.config.get("CELERY_RESULT_BACKEND"),
        broker=flask_app.config.get("CELERY_BROKER_URL"),
    )
    celery_instance.conf.update(flask_app.config)

    class ContextTask(celery_instance.Task):
        def __call__(self, *args, **kwargs):
            with flask_app.app_context():
                return self.run(*args, **kwargs)

    celery_instance.Task = ContextTask
    return celery_instance


app.config.update(
    CELERY_BROKER_URL=os.environ.get("CELERY_BROKER_URL"),
    CELERY_RESULT_BACKEND=os.environ.get("CELERY_RESULT_BACKEND"),
)
celery = make_celery(app)


def configure(flask_app):
    """flask app configuration function"""
    flask_app.config["SERVER_NAME"] = os.environ.get("FLASK_SERVER_NAME")
    flask_app.config["FLASK_DEBUG"] = os.environ.get("FLASK_DEBUG")
    flask_app.config["SWAGGER_UI_DOC_EXPANSION"] = os.environ.get(
        "RESTPLUS_SWAGGER_UI_DOC_EXPANSION"
    )
    flask_app.config["RESTPLUS_VALIDATE"] = os.environ.get("RESTPLUS_VALIDATE")
    flask_app.config["RESTPLUS_MASK_SWAGGER"] = os.environ.get("RESTPLUS_MASK_SWAGGER")
    flask_app.config["ERROR_404_HELP"] = os.environ.get("RESTPLUS_ERROR_404_HELP")


if __name__ == "__main__":
    from mongoengine import connect

    connect(
        host=os.environ.get("MONGO_DB_ENDPOINT"), db=os.environ.get("MONGO_DB_NAME")
    )
    configure(app)
    use_api(app)
    app.run()
