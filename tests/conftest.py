# pylint: disable-all
"""Fixtures used by pytest shared across all tests"""
import random
import uuid
from random import seed, randint

import pytest
import mongoengine
import torch
from mongoengine import disconnect, connect
from mongoengine import disconnect, connect
from pymongo import uri_parser

from recommender import create_app, User
from recommender.extensions import db
from recommender.models import Service, Recommendation
from tests.services.user_journey import UserJourney
from tests.factories.populate_database import populate_users_and_services
from tests.endpoints.conftest import mp_dump_data
from tests.engines.autoencoders.conftest import (
    ae_pipeline_config,
    mock_autoencoders_pipeline_exec,
    embedding_dims,
    embedding_exec,
)
from tests.engines.ncf.conftest import ncf_pipeline_config
from tests.engines.rl.conftest import (
    rl_pipeline_v1_config,
    rl_pipeline_v2_config,
    base_rl_pipeline_config,
)
from tests.endpoints.conftest import recommendation_data


@pytest.fixture()
def _app():
    """Create app"""
    return create_app()


@pytest.fixture
def client(_app):
    """Flask app client that you can make HTTP requests to"""
    yield _app.test_client()
    mongoengine.connection.disconnect_all()


@pytest.fixture
def original_mongo(_app):
    """MongoDB mock fixture
    Works wit multiproc/sequential tests but only with sequential code.
    Uses mongomock so it's very fast.

    WARNING:
        already it shouldn't be used because settings.TestingConfig.MONGODB_HOST doesn't use mongomock anymore.
    """

    with _app.app_context():
        yield db
        mongoengine.connection.disconnect_all()


@pytest.fixture
def singlemongo(_app):
    """MongoDB mock fixture.
    Works with sequential tests of the multicore code.

    Multicore code can't use mongomock so it uses real testing mongo db
    It can't be used with multicore tests (use sequential tests)
    """

    with _app.app_context():
        yield db
        mongoengine.connection.disconnect_all()


@pytest.fixture
def multimongo(_app):
    """MongoDB mock fixture.
    Works with multicore tests of the multicore code.

    Multicore code can't use mongomock so it uses real testing mongo db
    Multicore tests can't write to the same DB in the same time so this fixture create one mongo db instance for each test and it drops it on the teardown.
    """

    def _get_db_info(db):
        uri = db.app.config["MONGODB_HOST"]
        info_dict = uri_parser.parse_uri(uri)
        db_name = info_dict["database"]
        host = info_dict["nodelist"][0][0]
        port = info_dict["nodelist"][0][1]

        return db_name, host, port

    with _app.app_context():
        # alias = mongoengine.DEFAULT_CONNECTION_NAME = "default"
        # alias has to have above value because all mongoengine models in the recommender assume it.
        # Also connect and disconnect functions use "default" alias as a default argument

        db_name, host, port = _get_db_info(db)
        disconnect()
        testing_db_name = db_name + "_" + str(uuid.uuid4())  # TODO: maybe pid?!
        testing_db = connect(name=testing_db_name, host=host, port=port)
        yield db
        testing_db.drop_database(testing_db_name)
        disconnect()


@pytest.fixture
def fast_multimongo(_app):
    # TODO: maybe there is a way to force usage of mongomock in tests with no multiprocessing code
    # TODO: while using multimongo for tests with multiprocessing code in the same time.

    pass


mongo = multimongo


def users_services_args(valid=True):
    """Provide values for users and services generation"""
    seed(10)
    args = {
        "common_services_num": randint(40, 50),
        "unordered_services_num": randint(40, 50),
        "users_num": randint(1, 50),
        "k_common_services_min": randint(1, 3),
        "k_common_services_max": randint(4, 6),
        "verbose": False,
        "valid": valid,
    }

    return args


@pytest.fixture
def generate_users_and_services(mongo):
    """Generate valid users and services"""
    args = users_services_args(valid=True)
    populate_users_and_services(**args)


@pytest.fixture
def generate_invalid_users_and_services(mongo):
    """Generate invalid users and services"""
    args = users_services_args(valid=False)
    populate_users_and_services(**args)


@pytest.fixture
def delete_users_services():
    """Delete users and services"""
    User.drop_collection()
    Service.drop_collection()


@pytest.fixture
def generate_uas_and_recs(mongo):
    """Generate user actions and recommendations"""
    for i in range(2):
        if i % 2 == 0:
            anonymous = True
        else:
            anonymous = False
        j = UserJourney(anonymous=anonymous)
        r = Recommendation.objects(visit_id=j.lluj.last_visit_id).first()
        for _ in range(5):
            ua = j.lluj.click_random_ua_from_rec_tree(r, n=6)
            j.lluj.go_to_rec_page(ua)
            js = [j.service() for _ in range(3)]
            j2 = js[0].next(random.randint(1, 4)).order().go_to_panel()
            j3 = js[1].next(random.randint(1, 4)).go_to_panel()
            j.next(1).order().go_to_panel()
            j.next(2).order().next(1).go_to_panel()
            j.next(3).order().next(1)
        j.lluj.go_to_rec_page(start_ua=ua)
