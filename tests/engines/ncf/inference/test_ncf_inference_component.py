# pylint: disable-all

import pytest
from torch import Tensor

from recommender.engine.agents.pre_agent.models import NeuralColaborativeFilteringModel
from recommender.engine.preprocessing import (
    user_and_services_to_tensors,
    user_and_service_to_tensors,
    precalculate_tensors,
    precalc_users_and_service_tensors,
)
from recommender.engine.preprocessing.embedder import Embedder
from recommender.engine.preprocessing.preprocessing import NoPrecalculatedTensorsError
from recommender.engine.preprocessing.transformers import create_users_transformer
from recommender.engines.ncf.inference.ncf_inference_component import (
    NCFInferenceComponent,
)
from recommender.engines.ncf.ml_components.neural_collaborative_filtering import (
    NEURAL_CF,
    NeuralCollaborativeFilteringModel,
)
from recommender.errors import (
    InvalidRecommendationPanelIDError,
    NoSavedMLComponentError,
)
from recommender.engine.agents.panel_id_to_services_number_mapping import PANEL_ID_TO_K

from recommender.models import User, Service
from recommender.models.ml_component import MLComponent

from ..fixtures import (
    generate_data,
    mock_autoencoders_pipeline_exec,
    pipeline_config,
    mock_ncf_pipeline_exec,
)


def test_ncf_inference_component(
    mongo, generate_data, pipeline_config, mock_ncf_pipeline_exec
):
    # With-no-model case
    ncf_model = NeuralCollaborativeFilteringModel.load(version=NEURAL_CF)
    MLComponent.objects(version=NEURAL_CF).first().delete()

    with pytest.raises(NoSavedMLComponentError):
        K = 2
        ncf_inference_component = NCFInferenceComponent(K)

    # With-model case
    ncf_model.save(version=NEURAL_CF)
    for panel_id_version, K in list(PANEL_ID_TO_K.items()):
        ncf_inference_component = NCFInferenceComponent(K)
        user = User.objects.first()
        context = {"panel_id": panel_id_version, "search_data": {}, "user_id": user.id}

        services_ids_1 = ncf_inference_component(context)
        assert isinstance(
            ncf_inference_component.neural_cf_model, NeuralCollaborativeFilteringModel
        )

        assert isinstance(services_ids_1, list)
        assert len(services_ids_1) == PANEL_ID_TO_K.get(context["panel_id"])
        assert all([isinstance(service_id, int) for service_id in services_ids_1])

        services_ids_2 = ncf_inference_component(context)
        assert services_ids_1 == services_ids_2

    for panel_id_version, K in list(PANEL_ID_TO_K.items()):
        ncf_inference_component = NCFInferenceComponent(K)
        context = {"panel_id": panel_id_version, "search_data": {}, "user_id": -1}

        services_ids_1 = ncf_inference_component(context)
        assert isinstance(
            ncf_inference_component.neural_cf_model, NeuralCollaborativeFilteringModel
        )

        assert isinstance(services_ids_1, list)
        assert len(services_ids_1) == PANEL_ID_TO_K.get(context["panel_id"])
        assert all([isinstance(service_id, int) for service_id in services_ids_1])

        services_ids_2 = ncf_inference_component(context)
        assert services_ids_1 != services_ids_2

    for panel_id_version, K in list(PANEL_ID_TO_K.items()):
        ncf_inference_component = NCFInferenceComponent(K)
        context = {"panel_id": panel_id_version, "search_data": {}}

        services_ids_1 = ncf_inference_component(context)
        assert isinstance(
            ncf_inference_component.neural_cf_model, NeuralCollaborativeFilteringModel
        )

        assert isinstance(services_ids_1, list)
        assert len(services_ids_1) == PANEL_ID_TO_K.get(context["panel_id"])
        assert all([isinstance(service_id, int) for service_id in services_ids_1])

        services_ids_2 = ncf_inference_component(context)
        assert services_ids_1 != services_ids_2

    with pytest.raises(InvalidRecommendationPanelIDError):
        NCFInferenceComponent(K=-1)


def test_user_and_services_to_tensors_errors(mongo, generate_data):
    with pytest.raises(NoPrecalculatedTensorsError):
        u1 = User.objects[0]
        user_and_service_to_tensors(user=u1, service=u1.accessed_services[0])

    with pytest.raises(NoPrecalculatedTensorsError):
        u1 = User.objects[0]
        user_and_services_to_tensors(user=u1, services=u1.accessed_services)

    with pytest.raises(NoPrecalculatedTensorsError):
        u1 = User.objects[0]
        precalculate_tensors([u1], create_users_transformer())
        user_and_services_to_tensors(user=u1, services=u1.accessed_services)


def test_user_and_services_to_tensors(
    mongo, generate_data, mock_autoencoders_pipeline_exec
):
    # TODO: import below constants from autoencoders/embedders:
    USER = "user"
    SERVICE = "service"

    user_embedder = Embedder.load(USER)
    user_embedder(User.objects, use_cache=False, save_cache=True)

    service_embedder = Embedder.load(SERVICE)
    service_embedder(Service.objects, use_cache=False, save_cache=True)

    u1 = User.objects[0]

    (
        users_ids,
        users_tensor,
        services_ids,
        services_tensor,
    ) = user_and_service_to_tensors(user=u1, service=u1.accessed_services[0])

    assert isinstance(users_ids, Tensor)
    assert len(users_ids.shape) == 1
    assert users_ids.shape[0] == 1

    assert isinstance(users_tensor, Tensor)
    assert len(users_tensor.shape) == 2
    assert users_tensor.shape[0] == 1

    assert isinstance(services_ids, Tensor)
    assert len(services_ids.shape) == 1
    assert services_ids.shape[0] == 1

    assert isinstance(services_tensor, Tensor)
    assert len(services_tensor.shape) == 2
    assert services_tensor.shape[0] == 1

    (
        users_ids,
        users_tensor,
        services_ids,
        services_tensor,
    ) = user_and_services_to_tensors(user=u1, services=u1.accessed_services)

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
