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
        "engine_version": "pre_agent",
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
    _agent_init_mock = mocker.patch(
        "recommender.engine.agents.base_agent.BaseAgent.__init__"
    )
    agent_call_mock = mocker.patch(
        "recommender.engine.agents.base_agent.BaseAgent.call"
    )
    deserializer_mock = mocker.patch(
        "recommender.services.deserializer.Deserializer.deserialize_recommendation"
    )
    mocked_recommended_services = [1, 2, 3]
    agent_call_mock.return_value = mocked_recommended_services

    response = client.post(
        "/recommendations",
        data=json.dumps(recommendation_data),
        content_type="application/json",
    )

    deserializer_data = deepcopy(recommendation_data)
    deserializer_data["services"] = mocked_recommended_services

    agent_call_mock.assert_called_once_with(recommendation_data)
    deserializer_mock.assert_called_once_with(deserializer_data)
    assert response.get_json() == {"recommendations": mocked_recommended_services}
