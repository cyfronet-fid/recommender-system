# pylint: disable-all

import pytest
import random
from torch import Tensor

from recommender.engines.autoencoders.ml_components.embedder import (
    Embedder,
    USER_EMBEDDER,
    SERVICE_EMBEDDER,
)
from recommender.engines.autoencoders.training.data_preparation_step import (
    create_users_transformer,
    precalculate_tensors,
)
from recommender.engines.ncf.inference.ncf_inference_component import (
    NCFInferenceComponent,
)
from recommender.engines.ncf.inference.ncf_ranking_inference_component import (
    NCFRankingInferenceComponent,
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
    UserCannotBeIdentified,
)

from recommender.models import User, Service
from recommender.models.ml_component import MLComponent
from recommender.services.fts import retrieve_services_for_recommendation


@pytest.mark.parametrize("user_id_type", ["user_id", "aai_uid"])
def test_ncf_inference_component(
    mongo,
    generate_users_and_services,
    ncf_pipeline_config,
    mock_ncf_pipeline_exec,
    user_id_type,
):
    # With-no-model case
    ncf_model = NeuralCollaborativeFilteringModel.load(version=NEURAL_CF)
    MLComponent.objects(version=NEURAL_CF).first().delete()

    with pytest.raises(NoSavedMLComponentError):
        K = 2
        NCFInferenceComponent(K)

    candidates = [service.id for service in Service.objects]

    # With-model case
    ncf_model.save(version=NEURAL_CF)

    for panel_id_version, K in list(PANEL_ID_TO_K.items()):
        ncf_inference_component = NCFInferenceComponent(K)
        assert isinstance(ncf_inference_component, NCFInferenceComponent)
        assert isinstance(ncf_inference_component.engine_name, str)
        user = random.choice(list(User.objects))

        context = {
            "user_id": {
                "panel_id": panel_id_version,
                "candidates": candidates,
                "search_data": {},
                "user_id": user.id,
            },
            "aai_uid": {
                "panel_id": panel_id_version,
                "candidates": candidates,
                "search_data": {},
                "aai_uid": user.aai_uid,
            },
        }[user_id_type]

        services_ids_1, _, _ = ncf_inference_component(context)

        assert isinstance(
            ncf_inference_component.neural_cf_model, NeuralCollaborativeFilteringModel
        )

        assert isinstance(services_ids_1, list)
        assert len(services_ids_1) == PANEL_ID_TO_K.get(context["panel_id"])
        assert all([isinstance(service_id, int) for service_id in services_ids_1])
        assert all(service in candidates for service in services_ids_1)

        services_ids_2, _, _ = ncf_inference_component(context)
        assert services_ids_1 == services_ids_2

    with pytest.raises(InvalidRecommendationPanelIDError):
        NCFInferenceComponent(K=-1)


def test_user_cannot_be_identified(
    mongo, generate_users_and_services, ncf_pipeline_config, mock_ncf_pipeline_exec
):
    ncf_inference_component = NCFInferenceComponent(3)
    candidates = [service.id for service in Service.objects]

    context = {
        "panel_id": "v1",
        "candidates": candidates,
        "search_data": {},
    }

    with pytest.raises(UserCannotBeIdentified):
        ncf_inference_component(context)


def test_ncf_ranking_inference_component_returns_full_ranking(
    mongo, generate_users_and_services, ncf_pipeline_config, mock_ncf_pipeline_exec
):
    candidates = [service.id for service in Service.objects]

    for panel_id_version, K in list(PANEL_ID_TO_K.items()):
        ncf_ranking_inference_component = NCFRankingInferenceComponent(K)

        user = random.choice(list(User.objects))
        context = {
            "panel_id": panel_id_version,
            "candidates": candidates,
            "search_data": {},
            "user_id": user.id,
        }

        ranked_services_ids, _, _ = ncf_ranking_inference_component(context)

        assert isinstance(ranked_services_ids, list)
        assert all([isinstance(service_id, int) for service_id in ranked_services_ids])
        not_ranked_services_ids = [
            s.id
            for s in retrieve_services_for_recommendation(
                candidates, user.accessed_services
            )
        ]
        assert set(not_ranked_services_ids) == set(ranked_services_ids)


def test_user_and_services_to_tensors_errors(mongo, generate_users_and_services):
    with pytest.raises(NoPrecalculatedTensorsError):
        u1 = User.objects[0]
        NCFInferenceComponent.user_and_services_to_tensors(
            user=u1, services=u1.accessed_services
        )

    with pytest.raises(NoPrecalculatedTensorsError):
        u1 = User.objects[0]
        precalculate_tensors([u1], create_users_transformer())
        NCFInferenceComponent.user_and_services_to_tensors(
            user=u1, services=u1.accessed_services
        )


def test_user_and_services_to_tensors(
    mongo, generate_users_and_services, mock_autoencoders_pipeline_exec
):
    user_embedder = Embedder.load(USER_EMBEDDER)
    user_embedder(User.objects, use_cache=False, save_cache=True)

    service_embedder = Embedder.load(SERVICE_EMBEDDER)
    service_embedder(Service.objects, use_cache=False, save_cache=True)

    u1 = User.objects[0]

    (
        users_ids,
        users_tensor,
        services_ids,
        services_tensor,
    ) = NCFInferenceComponent.user_and_services_to_tensors(
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
