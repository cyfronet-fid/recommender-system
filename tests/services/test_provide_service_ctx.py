# pylint: disable-all

import pytest
from tests.conftest import users_services_args
from recommender.services.provide_service_ctx import get_all_collection_ids, service_ctx
from recommender.models import Service, User
from tests.endpoints.conftest import recommendation_data
from recommender.errors import ServicesContextNotProvidedError


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
    1) elastic_services provided, page_id different from /dashboard
    2) elastic_services NOT provided, page_id different from /dashboard
    3) elastic_services NOT provided, page_id == /dashboard
    """
    # 1)
    returned_dict = service_ctx(recommendation_data)
    assert returned_dict == recommendation_data

    # 2)
    del recommendation_data["elastic_services"]
    with pytest.raises(ServicesContextNotProvidedError):
        service_ctx(recommendation_data)

    # 3)
    recommendation_data.update({"page_id": "/dashboard"})
    returned_dict = service_ctx(recommendation_data)

    args = users_services_args()
    services_num = args["common_services_num"] + args["unordered_services_num"]

    assert returned_dict.get("elastic_services")
    assert len(returned_dict["elastic_services"]) == services_num
    assert all(
        [type(service_id) == int for service_id in returned_dict["elastic_services"]]
    )
