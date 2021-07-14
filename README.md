# EOSC Marketplace Recommender System
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

![Python-app workflow](https://github.com/cyfronet-fid/recommender-system/actions/workflows/python-app.yml/badge.svg)

EOSC Marketplace Recommender System uses **Deep Reinforcement Learning** to suggests relevant scientific services to appropriate researchers on the EOSC Marketplace portal.

## Architecture
Recommender system works as a microservice and exposes API to the Marketplace.

The inner structure can be described as two elements:
- web service part based on `Celery` and`Flask` with API created, documented and validated with `Flask_restx` and `Swagger`
- deep reinforcement learning part based on `Pytorch` and other ML libraries

## Development environment

### Requirements
- [git](https://git-scm.com/)
- [Python 3.7.x](https://www.python.org/downloads/release/python-370/)
- [Pipenv](https://pypi.org/project/pipenv/)
- [MongoDB](https://www.mongodb.com/)
- [redis](https://redis.io/)


All required project packages are listed in the pipfile. For their instalation look at the setup section.
If you want use GPU with pytorch you need CUDA capable device.

### Setup
1. Install git, python and pipenv
2. Clone this repository and go to its root directory
```bash
git clone https://github.com/cyfronet-fid/recommender-system.git
```
3. Install all required project packages by executing
```bash
pipenv --dev install
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
In the console you should see output similar to this:
```bash
 * Serving Flask flask_app "app" (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: off
 * Running on http://localhost:5000/ (Press CTRL+C to quit)
127.0.0.1 - - [29/user/2021 21:56:14] "GET /api/v1/doc/ HTTP/1.1" 200 -
127.0.0.1 - - [29/user/2021 21:56:14] "GET /api/v1/swagger.json HTTP/1.1" 200 -
127.0.0.1 - - [29/user/2021 21:56:14] "GET /swaggerui/favicon-32x32.png HTTP/1.1" 200 -

```

NOTE: You can customize flask host and flask port by using `FLASK_RUN_HOST` and `FLASK_RUN_PORT` [env](#env-variables) variables accordingly.

### Celery
To run background tasks you also need a celery worker running alongside your server. To run the worker:
```bash
export FLASK_ENV=development
pipenv run celery -A worker:app worker --loglevel=info
```

NOTE: Celery needs a running [redis](#redis) broker server in the background.

### Redis
NOTE: It is recommended for the developers to use docker-compose to run all the background servers
(see [docker](#Docker) section below).

Recommender system is running celery to execute background tasks in a queue.
As a backend we are using redis. In the development environment we are assuming that
the redis is running on `redis://localhost:6379`.

NOTE: You can customize your redis host url using `REDIS_HOST` [env](#env-variables) variable.

### Mongo
NOTE: It is recommended for the developers to use docker-compose to run all the background servers
(see [docker](#Docker) section below).

Install and start the mongodb server following the mongo installation instructions. It should be running on the default
url `mongodb://localhost:27017`.

NOTE: You can customize your mongodb host path in `MONGODB_HOST` [env](#env-variables) variable.

### API
You can interact with recommender system microservice using API available (by default) here: http://localhost:5000/

### Docker
To run all background servers needed for development (redis, mongodb) it is recommended that you use Docker:
```bash
docker-compose up
```
Mongo will be exposed and available on your host on `127.0.0.1:27017`, and redis on `127.0.0.1:6379`, although
you can change them using `MONGODB_HOST` and `REDIS_HOST` [env](#env-variables) variables accordingly.

NOTE: You still need to set up Flask server and celery worker as shown above. This is advantageous over the next option
because you can run pytest from your IDE, easily debug the application, easily restart the broken flask server and 
additionally you don't need to rebuild your docker image if your dependencies change.

For full-stack local development deployment use:
```bash
docker-compose -f docker-compose.yml -f development.yml up
```
This will build application images and run base flask development server on `127.0.0.1:5000` 
(you can customize flask port and host using [env](#env-variables) variables).
This command will also run celery worker, mongo and redis.
You can immediately change the server code without restarting the containers.

To run jupyter notebook server along with the application stack run:
```bash
docker-compose -f docker-compose.yml -f jupyter.yml up
```
NOTE: The url of the jupyter server will be displayed in the docker-compose output 
(default: `http://127.0.0.1:8888/?token=SOME_JUPYTER_TOKEN`) (you can customize jupyter port and host using [env](#env-variables) variables)

### Training
Currently GPU support is available only in the experimental jupyter notebook `neural_cf` that contains whole ML cycle:
- Datasets creation, preprocessing and inspections
- Models creation (Autoencoders, NCF)
- Trainings
- Evaluation
- Inferention

GPU support can be enabled using environmental variable TRAINING_DEVICE (look into [ENV variables](#env-variables) section) , but for now it doesn't work in the dev/test/prod environments due to the fact that celery uses `fork` rather than `spawn` multiprocessing method - it is incompatibile with `CUDA`.

For now the only trainable recommendation agent is the Pre-Agent.

It can be trained via training endpoint (used in production) or via training tasks.

Before training, the development database should be populated. It can be accomplished in two ways:
 - posting marketplace database dump to the `database_dumps` endpoint
 - loading it manually using `load_mp_dump` function

If you load dump manually, don't forget to run `tasks/neural_cf/real_data_preprocessing.py` to preprocess data (In case of loading dump via endpoint preprocessing is done automatically).

After populating database you can run training in following ways:
 - using `training` endpoint
 - using `tasks/neural_cf/real_data_training.py`

Training automatically:
 - creates dataset and split it into train/validation/test datasets 
 - saves datasets
 - creates new model
 - trains model and saves it
 - displays training statistics and plots.

Server Pre Agent will use last saved model for recommendations.

If you don't have data, but you want to see how training looks like, use `tasks/neural_cf/fake_data_training.py` task.
It prepares fake data and run training on it. All of it uses testing database but the final trained model and transformers (used for data preprocessing) are saved into development database, so it can be used for recommendations in development environment.

If you're interested in ML experiments related to model or training you can use `training_and_inferention_example.ipynb`.

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

### ENV variables
We are using .env to store instance specific constants or secrets. This file is not tracked by git and it needs to be 
present in the project root directory. Details:
- `MONGODB_HOST` - url and port of your running mongodb server (example: `127.0.0.1:27018`) or desired url and port of your mongodb
  server when ran using docker-compose (recommended)
- `REDIS_HOST` - url and port of your running redis server (example: `127.0.0.1:6380`) or desired url and port of your redis
  server when ran using docker-compose (recommended)
- `FLASK_RUN_HOST` - desired url of your application server (example: `127.0.0.1`)
- `FLASK_RUN_PORT` - desired port of your application server (example: `5001`)
- `JUPYTER_RUN_PORT` - desired port of your jupyter server when ran using docker (example: `8889`)
- `JUPYTER_RUN_HOST` - desired host of your jupyter server when ran using docker (example: `127.0.0.1`)
- `CELERY_LOG_LEVEL` - log level of your celery worker when ran using docker (one of: `CRITICAL`, `ERROR`, `WARN`, `INFO` or `DEBUG`)
- `SENTRY_DSN` -  The DSN tells the sentry where to send the events (example: `https://16f35998712a415f9354a9d6c7d096e6@o556478.ingest.sentry.io/7284791`). If that variable does not exist, sentry will just not send any events.
- `SENTRY_ENVIRONMENT` - environment name - it's optional and it can be a free-form string. If not specified and using docker, it is set to `development`/`testing`/`production` respectively to the docker environment.
- `SENTRY_RELEASE` - human readable release name - it's optional and it can be a free-form string. If not specified, sentry automatically set it based on commit revision number.
- `TRAINING_DEVICE` - the device used for training of neural networks: `cuda` for GPU support or `cpu` (note: `cuda` support is experimental and works only in jupyter notebook `neural_cf` - not in the recommender dev/prod/test environment)
- `AGENT_VERSION` - the version of the recommender agent (one of `pre_agent`, `rl_agent`) - Whenever request handling or celery task need this variable, it is dynamically loaded from the .env file, so you can change it during flask server runtime.

NOTE: All the above variables have reasonable defaults, so if you want you can just have your .env file empty.

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
While commiting using PyCharm Git GUI, pre-commit doesn't use project environment and can't find modules used in hooks.
To fix this, go to `.git/hooks/pre-commit` generated by the above command in project directory and replace:
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
`Sentry` is integrated with the `flask` server and the `celery` task queue manager so all unhandled exceptions from these entities will be tracked and sent to the sentry.
Customization of the sentry ntegration can be done vie environmental variables (look into [ENV variables](#env-variables) section) - more about them [here](https://docs.sentry.io/platforms/python/configuration/options/)