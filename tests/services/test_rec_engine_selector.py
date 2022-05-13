# pylint: disable-all

import pytest
from pytest import fixture
import os

from recommender.services.rec_engine_selector import (
    get_K,
    get_default_recommendation_alg,
    get_engine_names,
    engine_loader,
    load_engine,
    NCF_ENGINE_NAME,
    RL_ENGINE_NAME,
)
from recommender.engines.rl.inference.rl_inference_component import RLInferenceComponent
from recommender.engines.ncf.inference.ncf_inference_component import (
    NCFInferenceComponent,
)
from recommender.models.ml_component import MLComponent
from tests.endpoints.test_recommendations import recommendation_data
from tests.conftest import (
    generate_users_and_services,
)
from tests.engines.ncf.conftest import mock_ncf_pipeline_exec
from tests.engines.rl.conftest import mock_rl_pipeline_exec
from recommender.errors import (
    InvalidRecommendationPanelIDError,
    NoSavedMLComponentError,
)


@fixture
def get_engines():
    engines = {
        NCF_ENGINE_NAME: NCFInferenceComponent,
        RL_ENGINE_NAME: RLInferenceComponent,
    }
    return engines


def test_get_K(recommendation_data):
    """
    Expected behaviour:
    get_K function should map panel_id = {v1, v2} on {3, 2}
    """
    recommendation_data["panel_id"] = "v1"
    K = get_K(recommendation_data)
    assert K == 3

    recommendation_data["panel_id"] = "v2"
    K = get_K(recommendation_data)
    assert K == 2

    # Everything different from v1 and v2 should raise an error
    list_of_errors = [3, 2, "v3", "v0"]

    for error in list_of_errors:
        with pytest.raises(InvalidRecommendationPanelIDError):
            recommendation_data["panel_id"] = error
            get_K(recommendation_data)


def test_get_default_recommendation_alg():
    """
    Expected behaviour:
    DEFAULT_RECOMMENDATION_ALG=RL - return RL
    DEFAULT_RECOMMENDATION_ALG=NCF - return NCF
    Not set or something else - return RL
    """
    os.environ["DEFAULT_RECOMMENDATION_ALG"] = "RL"
    assert get_default_recommendation_alg() == "RL"

    os.environ["DEFAULT_RECOMMENDATION_ALG"] = "NCF"
    assert get_default_recommendation_alg() == "NCF"

    os.environ["DEFAULT_RECOMMENDATION_ALG"] = "NFC"
    assert get_default_recommendation_alg() == "RL"

    os.environ["DEFAULT_RECOMMENDATION_ALG"] = ""
    assert get_default_recommendation_alg() == "RL"


def test_get_engine_names(get_engines):
    """
    Expected behaviour:
    Order in which engine should be selected:
    1) Engine from a body request if this name is in engines_keys,
    2) Default engine if this name is in engines_keys,
    3) Any engine name that exists.
    """

    engine_from_req = ("RL", "NCF")
    default_engine = ("RL", "NCF")
    engines_keys = list(get_engines.keys())

    # 1. case - engine_from_req is specified, and engines_keys are the same as the range of engine_from_req
    for eg_req in engine_from_req:
        for eg_def in default_engine:
            engine_names = get_engine_names(eg_req, eg_def, engines_keys)
            expected_eg_names = ["RL", "NCF"] if eg_req == "RL" else ["NCF", "RL"]
            assert engine_names == expected_eg_names

    # 2. case - engine_from_req is NOT specified, and engines_keys are the same as the range of default_engine
    engine_from_req = None
    for eg_def in default_engine:
        engine_names = get_engine_names(engine_from_req, eg_def, engines_keys)
        expected_eg_names = ["RL", "NCF"] if eg_def == "RL" else ["NCF", "RL"]
        assert engine_names == expected_eg_names

    # 3. case - engine_from_req and default_engine are NOT specified
    engine_from_req = None
    default_engine = None
    engine_names = get_engine_names(engine_from_req, default_engine, engines_keys)
    expected_eg_names = [engines_keys[0], engines_keys[1]]
    assert engine_names == expected_eg_names

    # 4. case - engines_keys includes engines than engine_from_req nad default_engine
    engine_from_req = ("RL", "NCF")
    default_engine = ("RL", "NCF")
    # Last index of engines_keys, so it is expected to be the last element of returned names
    engines_keys.append("New Engine")

    for eg_req in engine_from_req:
        for eg_def in default_engine:
            engine_names = get_engine_names(eg_req, eg_def, engines_keys)
            expected_eg_names = (
                ["RL", "NCF", "New Engine"]
                if eg_req == "RL"
                else ["NCF", "RL", "New Engine"]
            )
            assert engine_names == expected_eg_names


def test_engine_loader(
    get_engines,
    generate_users_and_services,
    mock_rl_pipeline_exec,
    mock_ncf_pipeline_exec,
):
    """
    Expected behaviour:
    If there exists any saved engine, return the engine, else raise NoSavedMLComponentError
    """
    engine_names = ["RL", "NCF"]
    engines = get_engines
    K = 3

    # 1. case RL and NCF engines are saved to the DB
    engine, engine_name = engine_loader(engine_names, engines, K)
    assert type(engine) == RLInferenceComponent
    assert engine_name == "RL"

    engine_names = ["NCF", "RL"]
    engine, engine_name = engine_loader(engine_names, engines, K)
    assert type(engine) == NCFInferenceComponent
    assert engine_name == "NCF"

    # 2. case no engine is saved
    MLComponent.drop_collection()

    with pytest.raises(NoSavedMLComponentError):
        engine_loader(engine_names, engines, K)


def test_engine_loader2(
    get_engines, generate_users_and_services, mock_rl_pipeline_exec
):
    # 3. case NCF engine is requested but does not exist - RL engine should be returned
    engine_names = ["NCF", "RL"]
    engines = get_engines
    K = 3

    engine, engine_name = engine_loader(engine_names, engines, K)
    assert type(engine) == RLInferenceComponent
    assert engine_name == "RL"


def test_engine_loader3(
    get_engines,
    generate_users_and_services,
    mock_ncf_pipeline_exec,
):
    # 4. case RL engine is requested but does not exist - NCF engine should be returned
    engine_names = ["RL", "NCF"]
    engines = get_engines
    K = 3

    engine, engine_name = engine_loader(engine_names, engines, K)
    assert type(engine) == NCFInferenceComponent
    assert engine_name == "NCF"


def test_load_engine(
    generate_users_and_services,
    mock_rl_pipeline_exec,
    mock_ncf_pipeline_exec,
    recommendation_data,
):
    """
    Expected behaviour:
    Return engine
    """
    recommendation_data["engine_version"] = "NCF"
    engine, engine_name = load_engine(recommendation_data)
    assert type(engine) == NCFInferenceComponent
    assert engine_name == "NCF"

    recommendation_data["engine_version"] = "RL"
    engine, engine_name = load_engine(recommendation_data)
    assert type(engine) == RLInferenceComponent
    assert engine_name == "RL"
