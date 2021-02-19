#!/bin/bash
export FLASK_ENV=development
export FLASK_APP=app.py
pipenv run celery -A celery_worker:app worker --loglevel=info --detach
pipenv run flask run
