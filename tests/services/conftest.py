"""Fixtures for services testing"""
import pytest


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
            "categories": [1],
            "geographical_availabilities": ["PL"],
            "order_type": "open_access",
            "providers": [1],
            "rating": "5",
            "related_platforms": [1],
            "scientific_domains": [1],
            "sort": "_score",
            "target_users": [1],
        },
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
            "root": {"type": "recommendation_panel", "panel_id": "v1", "service_id": 1},
        },
        "target": {
            "visit_id": "9f543b80-dd5b-409b-a619-6312a0b04f4f",
            "page_id": "service_about",
        },
        "action": {"type": "button", "text": "Details", "order": True},
    }
