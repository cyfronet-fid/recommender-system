import os
import pytest

from pymodm.errors import DoesNotExist
from pymongo import MongoClient
from inflection import underscore, pluralize

from app.services.db_handler import DbHandler
from tests.test_helpers import pymodm_json_repr


class TestDbHandler:
    @pytest.fixture
    def mp_dump_data(self):
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

    @pytest.fixture
    def db_handler(self):
        yield DbHandler(os.environ["MONGO_DB_TEST_NAME"])
        # Explicit pymongo call to ensure db test drop
        MongoClient(os.environ["MONGO_DB_ENDPOINT"]).drop_database(
            os.environ["MONGO_DB_TEST_NAME"]
        )

    def test_load_and_drop(self, db_handler, mp_dump_data):
        db_handler.load(mp_dump_data)

        mongo_objects = {
            underscore(pluralize(model.__name__)): list(model.objects.all())
            for model in db_handler.mongo_model_classes
        }

        raw_mongo_objects = {
            k: [pymodm_json_repr(x) for x in v] for k, v in mongo_objects.items()
        }

        assert raw_mongo_objects == mp_dump_data

        # Some proper dereference model checks
        assert [
            pymodm_json_repr(x) for x in mongo_objects["services"][0].providers
        ] == raw_mongo_objects["providers"]
        assert [
            pymodm_json_repr(x) for x in mongo_objects["services"][0].categories
        ] == raw_mongo_objects["categories"]
        assert [
            pymodm_json_repr(x) for x in mongo_objects["users"][0].accessed_services
        ] == raw_mongo_objects["services"]

        db_handler.drop()

        for model in db_handler.mongo_model_classes:
            with pytest.raises(DoesNotExist):
                model.objects.all().first()
