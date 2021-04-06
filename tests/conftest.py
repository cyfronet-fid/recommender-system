# pylint: disable=fixme

"""Fixtures used by pytest shared across all tests"""

import pytest
import mongoengine

from recommender import create_app
from recommender.extensions import db


@pytest.fixture()
def _app():
    return create_app()


@pytest.fixture
def client(_app):
    """Flask app client that you can make HTTP requests to"""
    yield _app.test_client()
    mongoengine.connection.disconnect_all()


@pytest.fixture
def mongo(_app):
    """MongoDB mock fixture"""
    with _app.app_context():
        yield db
        mongoengine.connection.disconnect_all()


@pytest.fixture
def mp_dump_data():
    """Example MP database dump"""
    return {
        "services": [
            {
                "id": 1,
                "name": "test",
                "description": "desc",
                "tagline": "tag",
                "countries": ["PL", "US"],
                "rating": "2",
                "order_type": "open_access",
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
                "rating": "2",
                "order_type": "open_access",
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
def recommendation_json_dict():
    """Fixture of json dict of the recommendations endpoint request"""

    return {
        "user_id": 1,
        "unique_id": "5642c351-80fe-44cf-b606-304f2f338122",
        "timestamp": "2021-03-25T12:43:53.118Z",
        "visit_id": "202090a4-de4c-4230-acba-6e2931d9e37c",
        "page_id": "some_page_identifier",
        "panel_id": "v1",
        "services": [1, 2, 3],
        "search_data": {
            "q": "Cloud GPU",
            "category_id": 1,
            "geographical_availabilities": [
                "PL"
            ],
            "order_type": "open_access",
            "providers": [
                1
            ],
            "rating": "5",
            "related_platforms": [
                1
            ],
            "scientific_domains": [
                1
            ],
            "sort": "_score",
            "target_users": [
                1
            ]
        }
    }


@pytest.fixture
def user_action_json_dict():
    """Fixture of json dict of the user_actions endpoint request"""
    return {
        "user_id": 1,
        "unique_id": "5642c351-80fe-44cf-b606-304f2f338122",
        "timestamp": "2021-03-25T14:10:42.368Z",
        "source": {
            "visit_id": "202090a4-de4c-4230-acba-6e2931d9e37c",
            "page_id": "services_catalogue_list",
            "root": {
                "type": "recommendation_panel",
                "panel_id": "v1",
                "service_id": 1
            }
        },
        "target": {
            "visit_id": "9f543b80-dd5b-409b-a619-6312a0b04f4f",
            "page_id": "service_about"
        },
        "action": {
            "type": "button",
            "text": "Details",
            "order": True
        }
    }
