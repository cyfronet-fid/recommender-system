# pylint: disable-all

import pytest
from tests.conftest import users_services_args
from recommender.services.provide_ctx import get_all_collection_ids, service_ctx
from recommender.models import Service, User
from tests.endpoints.conftest import recommendation_data


def test_get_all_collection_ids(mongo, generate_users_and_services):
    """
    Check:
    - If it returns an object for all documents in the collection
    - If those objects are int type
    """
    # The expected number of users/services
    args = users_services_args()
    users_num = args["users_num"]
    services_num = args["common_services_num"] + args["unordered_services_num"]

    user_ids = get_all_collection_ids(User)
    service_ids = get_all_collection_ids(Service)

    assert len(user_ids) == users_num
    assert len(service_ids) == services_num
    assert all([type(user_id) == int for user_id in user_ids])
    assert all([type(service_id) == int for service_id in service_ids])


def test_service_ctx(mongo, generate_users_and_services, recommendation_data):
    """
    Check:
    K recommendations returned:
    1) candidates provided,
    2) candidates NOT provided, empty list
    3) candidates NOT provided, the key does not exist

    Sort by relevance:
    4) candidates NOT provided,
    5) candidates provided,

    """
    # All services from db
    args = users_services_args()
    all_services = args["common_services_num"] + args["unordered_services_num"]

    # 1)
    returned_dict = service_ctx(recommendation_data)
    assert returned_dict == recommendation_data

    # 2)
    recommendation_data["candidates"] = []
    candidates = service_ctx(recommendation_data)["candidates"]
    assert len(candidates) == all_services
    assert all([type(service_id) == int for service_id in candidates])

    # 3)
    del recommendation_data["candidates"]
    candidates = service_ctx(recommendation_data)["candidates"]
    assert len(candidates) == all_services
    assert all([type(service_id) == int for service_id in candidates])

    # 4)
    recommendation_data["engine_version"] = "NCFRanking"

    candidates = service_ctx(recommendation_data)["candidates"]
    assert len(candidates) == all_services
    assert all([type(service_id) == int for service_id in candidates])

    # 5)
    l = [1, 2, 3]
    recommendation_data["candidates"] = l

    candidates = service_ctx(recommendation_data)["candidates"]
    assert len(candidates) == len(l)
    assert all([type(service_id) == int for service_id in candidates])
