# EOSC Marketplace Recommender System
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

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
pipenv install
```

4. To open project virtual environment shell, type:
```bash
pipenv shell
```

### Launching

Launch EOSC Marketplace Recommender System with executing:
```
python run.py
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

### API
You can interact with recommender system microservice using API available (by deafult) here: http://localhost:5000/api/v1/doc/

### Application configuration
All configuration constants (e.g. app server name and port) are located in the `.env` file

### Pre-commit in PyCharm
While commiting using PyCharm Git GUI, pre-commit doesn't use project environment and can't find modules used in hooks.
To fix this, go to `.git/hooks/pre-commit` in project directory and replace:
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