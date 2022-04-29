# EOSC Marketplace Recommender System
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

![Python-app workflow](https://github.com/cyfronet-fid/recommender-system/actions/workflows/python-app.yml/badge.svg)

EOSC Marketplace Recommender System uses **Deep Reinforcement Learning** to suggest relevant scientific services to appropriate researchers on the EOSC Marketplace portal.

## Architecture
The recommender system works as a microservice and exposes API to the Marketplace.

The inner structure can be described as two elements:
- web service part based on `Celery` and`Flask` with API created, documented and validated with `Flask_restx` and `Swagger`
- deep reinforcement learning part based on `Pytorch` and other ML libraries

## Development environment

### Requirements
- [Git](https://git-scm.com/)
- [Python 3.10.x](https://www.python.org/downloads/release/python-3104/)
- [Pipenv](https://pypi.org/project/pipenv/)
- [MongoDB](https://www.mongodb.com/)
- [Redis](https://redis.io/)


All required project packages are listed in the `pipfile`. For their installation look at the [setup](#setup).
If you want to use GPU with PyTorch you need CUDA capable device.

### Setup
1. Install `git`, `python` and `pipenv`
2. Clone this repository and go to its root directory
```bash
git clone https://github.com/cyfronet-fid/recommender-system.git
```
3. Install all required project packages by executing
```bash
pipenv install --dev
```

4. To open project virtual environment shell, type:
```bash
pipenv shell
```

### Server

Launch EOSC Marketplace Recommender server by executing in the project root directory:
```
export FLASK_ENV=development
export FLASK_APP=app.py
pipenv run flask run
```
NOTE: You can customize flask host and flask port by using `FLASK_RUN_HOST` and `FLASK_RUN_PORT` [env](#env-variables) variables accordingly.

### Celery
To run background tasks you also need a celery worker running alongside your server. To run the worker:
```bash
export FLASK_ENV=development
pipenv run celery -A worker:app worker
```

NOTE: Celery needs a running [redis](#redis) broker server in the background.

### Redis
NOTE: It is recommended for the developers to use docker-compose to run all the background servers
(see [docker](#Docker) section below).

The recommender system is running celery to execute background tasks in a queue.
As a backend, we are using Redis. By default, Redis is running on `redis://localhost:6379`.

NOTE: You can customize your Redis host URL using `REDIS_HOST` [env](#env-variables) variable.

### Mongo
NOTE: It is recommended for the developers to use docker-compose to run all the background servers
(see [docker](#Docker) section below).

Install and start the MongoDB server following the Mongo installation instructions. It should be running on the default
URL `mongodb://localhost:27017`.

NOTE: You can customize your MongoDB host path in the `MONGODB_HOST` [env](#env-variables) variable.

### API
You can interact with recommender system microservice using API available (by default) here: http://localhost:5000/

### Docker
To run all background servers needed for development (`Redis`, `MongoDB`) it is recommended that you use Docker:
```bash
docker-compose up
```
Mongo will be exposed and available on your host on `127.0.0.1:27017`, and Redis on `127.0.0.1:6379`, although
you can change them using `MONGODB_HOST` and `REDIS_HOST` [env](#env-variables) variables accordingly.

NOTE: You still need to set up Flask server and Celery worker as shown above. This is advantageous over the next option
because you can run Pytest directly from your IDE, debug the application simply, restart Flask server easily, 
and you also avoid having to rebuild your docker image if your dependencies change.

For full-stack local development deployment use:
```bash
docker-compose -f docker-compose.yml -f development.yml up
```
This will build application images and run the base Flask development server on `127.0.0.1:5000` 
(you can customize flask port and host using [env](#env-variables) variables).
This command will also run Celery worker, Mongo and Redis.
You can immediately change the server code without restarting the containers.

To run the Jupyter notebook server along with the application stack run:
```bash
docker-compose -f docker-compose.yml -f jupyter.yml up
```
NOTE: The URL of the Jupyter server will be displayed in the docker-compose output 
(default: `http://127.0.0.1:8888/?token=SOME_JUPYTER_TOKEN`) (you can customize Jupyter port and host using [env](#env-variables) variables)

### Tests
To run all the tests in our app run:
```bash
export FLASK_ENV=testing
pipenv run pytest ./tests
```
...or you can run them using docker:
```bash
docker-compose -f docker-compose.testing.yml up && docker-compose -f docker-compose.testing.yml down
```

### Training

Recommender system can use one of two recommendation engines implemented:
- `NCF` - based on [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031) paper.
- `RL` - based on [Deep Deterministic Policy Gradient](https://arxiv.org/abs/1509.02971) paper.

There are two ways in which the recommender system can be trained. 
1) First method is to send a database dump to the RS `/update` endpoint.
It may be done, for example, by triggering `./bin/rails recommender:update` task on the Marketplace side. It sends a database dump to the `/update` endpoint of the Recommender System. It sends the most current training data, preprocesses it, and utilizes it to train the models that are required.

2) The second method is to use Flask commands: 
- `flask train all` - the equivalent of training via endpoint `/update` - triggers the training of each pipeline,
- `flask train ae` - triggers the training of autoencoders pipeline,
- `flask train embedding` - triggers the training of embeddings,
- `flask train ncf` - triggers the training of NCF pipeline (provides 3 recommendations),
- `flask train rl` - triggers the training of RL pipeline (provides 3 recommendations),

GPU support can be enabled using an environmental variable `TRAINING_DEVICE` (look into [ENV variables](#env-variables) section).

After training is finished, the system is immediately ready for serving recommendations (no manual reloading is needed).

To specify from which engine the recommendations are requested, provide an optional `engine_version` parameter inside the body of `\recommendations` endpoint. `NCF` denotes the NCF engine, while `RL` indicates the RL engine.
It is possible to define which algorithm should be used by default in the absence of the `engine_version` parameter by modifying the `DEFAULT_RECOMMENDATION_ALG` parameter from .env file
(look into [ENV variables](#env-variables) section).

### Generating DB entries for development
Our recommender, like other systems, requires data to perform properly. Several prepared commands can be used to generate such data:
- `flask db seed` - it allows to seed a database with any number of synthetic users and services. The exact number can be adjusted here [seed](https://github.com/cyfronet-fid/recommender-system/blob/040a41725f7a1f5ef1a7ea060744a18cd0b6fc7a/recommender/commands/db.py#L29),
- `flask db seed_faker` - analysis the users and services from a current database and produces some documents which later on will enable to generate more realistic synthetic users and services,
- `flask db drop_mp` - drops the documents from the RS database which were sent by the MP database dump,
- `flask db drop_models` - drops machine learning models from m_l_component collection,
- `flask db regenerate_sarses` - based on new user actions - add new SARSes and regenerate existing ones that are deprecated.

### Migrations
We are using MongoDB as our database, which is a NoSQL, schema-less, document-based DB. However, we are also using `mongoengine` - an
ODM (Object Document Mapping), which defines a "schema" for each document (like specifying field names or required values). 
This means that we need a minimalistic migration system to apply the defined "schema" changes, 
like changing a field name or dropping a collection, if we want to maintain the Application <=> DB integrity.

Migration flask CLI commands (first set the `FLASK_ENV` variable to either `development` or `production`):
- `flask migrate apply` - applies migrations that have not been applied yet
- `flask migrate rollback` - reverts the previously applied migration
- `flask migrate list` - lists all migrations along with their application status
- `flask migrate check` - checks the integrity of the migrations - if the migration files match the DB migration cache
- `flask migrate repopulate` - deletes migration cache and repopulates it with all the migrations defined in `/recommender/migrate` dir.

To create a new migration:
1. In the `/recommender/migrations` dir:
2. Create a python module with a name `YYYYMMDDMMHHSS_migration_name` (e.g. `20211126112815_remove_unused_collections`)
3. In this module create a migration class (with an arbitrary name) which inherits from `BaseMigration`
4. Implement `up` (application) and `down` (teardown) methods, by using `self.pymongo_db` ([pymongo](https://pymongo.readthedocs.io/en/stable/), a low-level adapter for mongoDB, connected to proper (dependent on the `FLASK_ENV` variable) recommender DB instance)

(See existing files in the `/recommender/migrate` dir for a more detailed example.)

**DO NOT DELETE EXISTING MIGRATION FILES. DO NOT CHANGE EXISTING MIGRATION FILE NAMES. DO NOT MODIFY THE CODE OF EXISTING MIGRATION FILES**

(If you performed any of those actions, run `flask migrate check` to determine what went wrong.)

### Documentation
The essential components of the recommendation system are also documented in our repository:
- [Data](https://github.com/cyfronet-fid/recommender-system/blob/main/docs/data.md),
- [Training](https://github.com/cyfronet-fid/recommender-system/blob/main/docs/training.md),
- [Evaluation](https://github.com/cyfronet-fid/recommender-system/blob/main/docs/evaluation.md),
- [User session gathering](https://github.com/cyfronet-fid/recommender-system/blob/main/docs/session_gathering/session_gathering.md)

### ENV variables
We are using .env to store instance-specific constants or secrets. This file is not tracked by git and it needs to be 
present in the project root directory. Details:
- `MONGODB_HOST` - URL and port of your running MongoDB server (example: `127.0.0.1:27018`) or desired URL and port of your MongoDB
  server when it is run using docker-compose (recommended)
- `REDIS_HOST` - URL and port of your running Redis server (example: `127.0.0.1:6380`) or desired URL and port of your Redis
  server when it is run using docker-compose (recommended)
- `FLASK_RUN_HOST` - desired URL of your application server (example: `127.0.0.1`)
- `FLASK_RUN_PORT` - desired port of your application server (example: `5001`)
- `JUPYTER_RUN_PORT` - desired port of your Jupyter server when ran using Docker (example: `8889`)
- `JUPYTER_RUN_HOST` - desired host of your Jupyter server when ran using Docker (example: `127.0.0.1`)
- `CELERY_LOG_LEVEL` - log level of your Celery worker when ran using Docker (one of: `CRITICAL`, `ERROR`, `WARN`, `INFO` or `DEBUG`)
- `SENTRY_DSN` -  The DSN tells the Sentry where to send the events (example: `https://16f35998712a415f9354a9d6c7d096e6@o556478.ingest.sentry.io/7284791`). If that variable does not exist, Sentry will just not send any events.
- `SENTRY_ENVIRONMENT` - environment name - it's optional and it can be a free-form string. If not specified and using Docker, it is set to `development`/`testing`/`production` respectively to the docker environment.
- `SENTRY_RELEASE` - human-readable release name - it's optional and it can be a free-form string. If not specified, Sentry automatically set it based on the commit revision number.
- `TRAINING_DEVICE` - the device used for training of neural networks: `cuda` for GPU support or `cpu` (note: `cuda` support is experimental and works only in Jupyter notebook `neural_cf` - not in the recommender dev/prod/test environment)
- `DEFAULT_RECOMMENDATION_ALG` - the version of the recommender engine (one of `NCF`, `RL`) - Whenever request handling or celery task need this variable, it is dynamically loaded from the .env file, so you can change it during flask server runtime.
- `RS_SUBSCRIBER_HOST` - the address of your JMS provider (optional)
- `RS_SUBSCRIBER_PORT` - the port of your JMS provider (optional)
- `RS_SUBSCRIBER_USERNAME` - your login to the JMS provider (optional)
- `RS_SUBSCRIBER_PASSWORD` - your password to the JMS provider (optional)
- `RS_SUBSCRIBER_TOPIC` - topic on which subscriber listens to jms (optional)
- `RS_SUBSCRIBER_SUBSCRIPTION_ID` - subscription id of the jms subscriber (optional)
- `RS_SUBSCRIBER_SSL` - whether to use ssl when connecting to jms (optional) (accepted values `0` or `1`, `yes` or `no`)
- `TEST_RS_SUBSCRIBER_HOST` - same as `RS_SUBSCRIBER_HOST` but used when testing via `pytest`  (default: `127.0.0.1`)
- `TEST_RS_SUBSCRIBER_PORT` - same as `RS_SUBSCRIBER_PORT` but used when testing via `pytest` (default: `61613`)
- `TEST_RS_SUBSCRIBER_USERNAME` - same as `RS_SUBSCRIBER_USERNAME` but used when testing via `pytest` (default: `guest`)
- `TEST_RS_SUBSCRIBER_PASSWORD` - same as `RS_SUBSCRIBER_PASSWORD` but used when testing via `pytest` (default: `guest`)
- `TEST_RS_SUBSCRIBER_TOPIC` - same as `RS_SUBSCRIBER_TOPIC` but used when testing via `pytest` (default: `topic/user_actions_test`)

NOTE: All the above variables have reasonable defaults, so if you want you can just have your .env file empty.

### JMS Subscriber

There is flask cli command to run JMS subscription which connects to databus and consumes user actions. It can be run
with following command

```shell
flask subscribe --host 127.0.0.1 --port 61613 --username guest --password guest 
```

For all available options run

```shell
flask subscribe --help
```

All arguments to subscribe can be read from environmental variables (see section about env variables above)


### Pre-commit
To activate pre-commit run:
```bash
pipenv run pre-commit install
```

### PyCharm Integrations
#### .env
Install [EnvFile plugin](https://plugins.jetbrains.com/plugin/7861-envfile). Go to the run configuration of your choice, switch to `EnvFile` tab, check `Enable EnvFile`, click `+` button below, select `.env` file and click `Apply` (Details on the plugin's page)

#### PyTest
In Pycharm, go to `Settings` -> `Tools` -> `Python Integrated Tools` -> `Testing` and choose `pytest`
Remember to put FLASK_ENV=testing env variable in the configuration.

#### Pre-commit
While committing using PyCharm Git GUI, pre-commit doesn't use project environment and can't find modules used in hooks.
To fix this, go to `.git/hooks/pre-commit` generated by the above command in the project directory and replace:
```python
# start templated
INSTALL_PYTHON = 'PATH/TO/YOUR/ENV/EXECUTABLE'
```

with:
```python
# start templated
INSTALL_PYTHON = 'PATH/TO/YOUR/ENV/EXECUTABLE'
os.environ['PATH'] = f'{os.path.dirname(INSTALL_PYTHON)}{os.pathsep}{os.environ["PATH"]}'
```

### External tools integration

#### Sentry
`Sentry` is integrated with the `Flask` server and the `Celery` task queue manager so all unhandled exceptions from these entities will be tracked and sent to the sentry.
Customization of the sentry integration can be done vie environmental variables (look into [ENV variables](#env-variables) section) - you can read more about them [here](https://docs.sentry.io/platforms/python/configuration/options/)
