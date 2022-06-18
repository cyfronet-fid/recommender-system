# pylint: disable-all

import pytest
import random
from torch import Tensor

from recommender.engines.ncf.inference.ncf_inference_component import (
    NCFInferenceComponent,
)
from recommender.engines.ncf.ml_components.neural_collaborative_filtering import (
    NEURAL_CF,
    NeuralCollaborativeFilteringModel,
)
from recommender.engines.panel_id_to_services_number_mapping import PANEL_ID_TO_K
from recommender.errors import (
    InvalidRecommendationPanelIDError,
    NoSavedMLComponentError,
    NoPrecalculatedTensorsError,
)

from recommender.models import User, Service
from recommender.models.ml_component import MLComponent


def test_ncf_inference_component(
    mongo, generate_users_and_services, ncf_pipeline_config, mock_ncf_pipeline_exec
):
    # With-no-model case
    ncf_model = NeuralCollaborativeFilteringModel.load(version=NEURAL_CF)
    MLComponent.objects(version=NEURAL_CF).first().delete()

    with pytest.raises(NoSavedMLComponentError):
        K = 2
        NCFInferenceComponent(K)

    elastic_services = [service.id for service in Service.objects]

    # With-model case
    ncf_model.save(version=NEURAL_CF)

    for panel_id_version, K in list(PANEL_ID_TO_K.items()):
        ncf_inference_component = NCFInferenceComponent(K)
        assert isinstance(ncf_inference_component, NCFInferenceComponent)
        assert isinstance(ncf_inference_component.engine_name, str)
        user = random.choice(list(User.objects))

        context = {
            "panel_id": panel_id_version,
            "elastic_services": elastic_services,
            "search_data": {},
            "user_id": user.id,
        }

        services_ids_1 = ncf_inference_component(context)

        assert isinstance(
            ncf_inference_component.neural_cf_model, NeuralCollaborativeFilteringModel
        )

        assert isinstance(services_ids_1, list)
        assert len(services_ids_1) == PANEL_ID_TO_K.get(context["panel_id"])
        assert all([isinstance(service_id, int) for service_id in services_ids_1])
        assert all(service in elastic_services for service in services_ids_1)

        services_ids_2 = ncf_inference_component(context)
        assert services_ids_1 == services_ids_2

    with pytest.raises(InvalidRecommendationPanelIDError):
        NCFInferenceComponent(K=-1)


def test_user_and_services_to_tensors(
    mongo, generate_users_and_services, mock_ncf_pipeline_exec
):

    u1 = User.objects[0]

    ncf_inference_component = NCFInferenceComponent(2)

    (
        users_ids,
        users_tensor,
        services_ids,
        services_tensor,
    ) = ncf_inference_component.user_and_services_to_tensors(
        user=u1, services=u1.accessed_services
    )

    assert isinstance(users_ids, Tensor)
    assert len(users_ids.shape) == 1
    assert users_ids.shape[0] == len(u1.accessed_services)

    assert isinstance(users_tensor, Tensor)
    assert len(users_tensor.shape) == 2
    assert users_tensor.shape[0] == len(u1.accessed_services)

    assert isinstance(services_ids, Tensor)
    assert len(services_ids.shape) == 1
    assert services_ids.shape[0] == len(u1.accessed_services)

    assert isinstance(services_tensor, Tensor)
    assert len(services_tensor.shape) == 2
    assert services_tensor.shape[0] == len(u1.accessed_services)
