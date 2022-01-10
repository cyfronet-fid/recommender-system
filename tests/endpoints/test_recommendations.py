# pylint: disable-all

import json
from copy import deepcopy

from pytest import fixture


@fixture
def recommendation_data():
    return {
        "user_id": 1,
        "unique_id": "5642c351-80fe-44cf-b606-304f2f338122",
        "timestamp": "2021-03-18T18:49:55.006Z",
        "visit_id": "202090a4-de4c-4230-acba-6e2931d9e37c",
        "page_id": "some_page_identifier",
        "panel_id": "v1",
        "engine_version": "NCF",
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


def test_recommendations(client, mocker, recommendation_data):
    _inference_component_init_mock = mocker.patch(
        "recommender.engines.base.base_inference_component.BaseInferenceComponent.__init__"
    )
    inference_component_call_mock = mocker.patch(
        "recommender.engines.base.base_inference_component.BaseInferenceComponent.__call__"
    )
    deserializer_mock = mocker.patch(
        "recommender.services.deserializer.Deserializer.deserialize_recommendation"
    )
    mocked_recommended_services = [1, 2, 3]
    inference_component_call_mock.return_value = mocked_recommended_services

    response = client.post(
        "/recommendations",
        data=json.dumps(recommendation_data),
        content_type="application/json",
    )

    deserializer_data = deepcopy(recommendation_data)
    deserializer_data["services"] = mocked_recommended_services

    inference_component_call_mock.assert_called_once_with(recommendation_data)
    deserializer_mock.assert_called_once_with(deserializer_data)
    assert response.get_json() == {"recommendations": mocked_recommended_services}
