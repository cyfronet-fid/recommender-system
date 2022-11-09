# pylint: disable-all

import json
from copy import deepcopy

import pytest

from recommender.engines.explanations import Explanation


@pytest.mark.parametrize(
    "recc_data_fixture", ["recommendation_data", "recommendation_data_with_aai_uid"]
)
def test_recommendations(client, mocker, recc_data_fixture, request):
    recc_data = request.getfixturevalue(recc_data_fixture)

    _inference_component_init_mock = mocker.patch(
        "recommender.engines.base.base_inference_component.MLEngineInferenceComponent.__init__"
    )
    inference_component_call_mock = mocker.patch(
        "recommender.engines.base.base_inference_component.MLEngineInferenceComponent.__call__"
    )
    deserializer_mock = mocker.patch(
        "recommender.services.deserializer.Deserializer.deserialize_recommendation"
    )
    send_recommendation_to_databus_mock = mocker.patch(
        "recommender.tasks.send_recommendation_to_databus.delay"
    )

    mock_recommended_services_ids = [1, 2, 3]
    mock_scores = [0.7, 0.2, 0.1]
    mock_explanations = 3 * [Explanation(long="mock_long", short="mock_short")]
    inference_component_call_mock.return_value = (
        mock_recommended_services_ids,
        mock_scores,
        mock_explanations,
    )

    response = client.post(
        "/recommendations",
        data=json.dumps(recc_data),
        content_type="application/json",
    )

    deserializer_data = deepcopy(recc_data)
    deserializer_data["services"] = mock_recommended_services_ids

    inference_component_call_mock.assert_called_once_with(recc_data)
    deserializer_mock.assert_called_once_with(deserializer_data)

    mock_explanations_long, mock_explanations_short = [
        list(t) for t in list(zip(*[(e.long, e.short) for e in mock_explanations]))
    ]

    assert response.get_json() == {
        "panel_id": recc_data["panel_id"],
        "recommendations": mock_recommended_services_ids,
        "explanations": mock_explanations_long,
        "explanations_short": mock_explanations_short,
        "scores": mock_scores,
        "engine_version": recc_data["engine_version"],
    }

    # AAI uid fixture contains client_id != marketplace, so it shouldn't create a celery task
    if recc_data_fixture == "recommendation_data_with_aai_uid":
        send_recommendation_to_databus_mock.assert_not_called()
    else:
        send_recommendation_to_databus_mock.assert_called_once_with(
            recc_data, response.get_json()
        )
