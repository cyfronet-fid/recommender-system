# pylint: disable-all

import json
from copy import deepcopy

from recommender.engines.explanations import Explanation


def test_recommendations(
    client, mocker, recommendation_data, recommendation_data_with_aai_uid
):

    for recc_data in [recommendation_data, recommendation_data_with_aai_uid]:
        _inference_component_init_mock = mocker.patch(
            "recommender.engines.base.base_inference_component.MLEngineInferenceComponent.__init__"
        )
        inference_component_call_mock = mocker.patch(
            "recommender.engines.base.base_inference_component.MLEngineInferenceComponent.__call__"
        )
        deserializer_mock = mocker.patch(
            "recommender.services.deserializer.Deserializer.deserialize_recommendation"
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
