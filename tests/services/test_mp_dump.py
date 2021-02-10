# pylint: disable-all

import pytest
from mongoengine import connect, disconnect
from inflection import underscore, pluralize

from app.db.mongo_models import MP_DUMP_MODEL_CLASSES
from app.services.mp_dump import load_mp_dump, drop_mp_dump
from tests.test_helpers import mongo_model_to_json


@pytest.fixture
def mp_dump_data():
    return {
        "services": [
            {
                "id": 1,
                "name": "test",
                "description": "desc",
                "tagline": "tag",
                "countries": ["PL", "US"],
                "categories": [1, 2],
                "providers": [1, 2],
                "resource_organisation": 1,
                "scientific_domains": [1, 2],
                "platforms": [1, 2],
                "target_users": [1, 2],
                "related_services": [2],
                "required_services": [2],
                "access_types": [1, 2],
                "access_modes": [1, 2],
                "trls": [1, 2],
                "life_cycle_statuses": [1, 2],
            },
            {
                "id": 2,
                "name": "test2",
                "description": "desc",
                "tagline": "tag",
                "countries": ["PL"],
                "categories": [2],
                "providers": [2],
                "resource_organisation": 2,
                "scientific_domains": [2],
                "platforms": [2],
                "target_users": [2],
                "related_services": [1],
                "required_services": [],
                "access_types": [2],
                "access_modes": [2],
                "trls": [2],
                "life_cycle_statuses": [2],
            },
        ],
        "users": [
            {
                "id": 1,
                "scientific_domains": [1, 2],
                "categories": [1, 2],
                "accessed_services": [1, 2],
            }
        ],
        "categories": [{"id": 1, "name": "c1"}, {"id": 2, "name": "c2"}],
        "providers": [{"id": 1, "name": "p1"}, {"id": 2, "name": "p2"}],
        "scientific_domains": [{"id": 1, "name": "sd1"}, {"id": 2, "name": "sd2"}],
        "platforms": [{"id": 1, "name": "pl1"}, {"id": 2, "name": "pl2"}],
        "target_users": [
            {"id": 1, "name": "tu1", "description": "desc"},
            {"id": 2, "name": "tu2", "description": "desc"},
        ],
        "access_modes": [
            {"id": 1, "name": "am1", "description": "desc"},
            {"id": 2, "name": "am2", "description": "desc"},
        ],
        "access_types": [
            {"id": 1, "name": "at1", "description": "desc"},
            {"id": 2, "name": "at2", "description": "desc"},
        ],
        "trls": [
            {"id": 1, "name": "trl-1", "description": "desc"},
            {"id": 2, "name": "trl-2", "description": "desc"},
        ],
        "life_cycle_statuses": [
            {"id": 1, "name": "lcs1", "description": "desc"},
            {"id": 2, "name": "lcs2", "description": "desc"},
        ],
    }


@pytest.fixture(scope="function")
def mongo():
    conn = connect(db="test", host="mongomock://localhost")
    yield conn
    conn.drop_database("test")
    conn.close()


def test_load_and_drop_mp_dump(mongo, mp_dump_data):
    load_mp_dump(mp_dump_data)

    mongo_objects = {
        underscore(pluralize(model.__name__)): list(model.objects.all())
        for model in MP_DUMP_MODEL_CLASSES
    }

    raw_mongo_objects = {
        k: [mongo_model_to_json(x) for x in v] for k, v in mongo_objects.items()
    }

    assert raw_mongo_objects == mp_dump_data

    # Some proper dereference model checks
    assert [
        mongo_model_to_json(x) for x in mongo_objects["services"][0].providers
    ] == raw_mongo_objects["providers"]
    assert [
        mongo_model_to_json(x) for x in mongo_objects["services"][0].categories
    ] == raw_mongo_objects["categories"]
    assert [
        mongo_model_to_json(x) for x in mongo_objects["users"][0].accessed_services
    ] == raw_mongo_objects["services"]

    drop_mp_dump()

    for model in MP_DUMP_MODEL_CLASSES:
        assert model.objects.first() is None
