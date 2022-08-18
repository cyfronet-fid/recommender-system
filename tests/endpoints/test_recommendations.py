# pylint: disable-all

import json
from copy import deepcopy


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
        mocked_recommended_services = [1, 2, 3]
        inference_component_call_mock.return_value = mocked_recommended_services

        response = client.post(
            "/recommendations",
            data=json.dumps(recc_data),
            content_type="application/json",
        )

        deserializer_data = deepcopy(recc_data)
        deserializer_data["services"] = mocked_recommended_services

        inference_component_call_mock.assert_called_once_with(recc_data)
        deserializer_mock.assert_called_once_with(deserializer_data)
        assert response.get_json() == {"recommendations": mocked_recommended_services}
