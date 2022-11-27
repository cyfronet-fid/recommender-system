# pylint: disable-all

import pytest
import os

from recommender.services.engine_selector import (
    get_K,
    get_default_recommendation_alg,
    get_engine_names,
    engine_loader,
    load_engine,
)
from recommender.engines.rl.inference.rl_inference_component import (
    RLInferenceComponent,
)
from recommender.engines.ncf.inference.ncf_inference_component import (
    NCFInferenceComponent,
)
from recommender.engines.random.inference.random_inference_component import (
    RandomInferenceComponent,
)
from recommender.models.ml_component import MLComponent
from tests.endpoints.conftest import (
    recommendation_data,
    recommendation_data_with_aai_uid,
)
from tests.conftest import (
    generate_users_and_services,
)
from tests.engines.ncf.conftest import mock_ncf_pipeline_exec
from tests.engines.rl.conftest import mock_rl_pipeline_exec
from recommender.errors import (
    InvalidRecommendationPanelIDError,
    NoSavedMLComponentError,
)


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


def test_get_default_recommendation_alg(get_engines):
    """
    Expected behaviour:
    DEFAULT_RECOMMENDATION_ALG in (RL, rl, Rl, rL) - return RL
    DEFAULT_RECOMMENDATION_ALG in (NCF, ncf, Ncf...) - return NCF
    DEFAULT_RECOMMENDATION_ALG in (random, Random, ...) - return random
    Not set or something else - return RL
    """
    os.environ["DEFAULT_RECOMMENDATION_ALG"] = "RL"
    assert get_default_recommendation_alg(get_engines.keys()) == "RL"
    os.environ["DEFAULT_RECOMMENDATION_ALG"] = "NCF"
    assert get_default_recommendation_alg(get_engines.keys()) == "NCF"
    os.environ["DEFAULT_RECOMMENDATION_ALG"] = "Random"
    assert get_default_recommendation_alg(get_engines.keys()) == "Random"

    os.environ["DEFAULT_RECOMMENDATION_ALG"] = "NFC"
    assert get_default_recommendation_alg(get_engines.keys()) == "RL"
    os.environ["DEFAULT_RECOMMENDATION_ALG"] = ""
    assert get_default_recommendation_alg(get_engines.keys()) == "RL"

    os.environ["DEFAULT_RECOMMENDATION_ALG"] = "rl"
    assert get_default_recommendation_alg(get_engines.keys()) == "RL"
    os.environ["DEFAULT_RECOMMENDATION_ALG"] = "ncf"
    assert get_default_recommendation_alg(get_engines.keys()) == "NCF"
    os.environ["DEFAULT_RECOMMENDATION_ALG"] = "random"
    assert get_default_recommendation_alg(get_engines.keys()) == "Random"


def test_get_engine_names(get_engines):
    """
    Expected behaviour:
    Order in which engine should be selected:
    1) Engine from a body request if this name is in engines_keys,
    2) Default engine if this name is in engines_keys,
    3) Any engine name that exists.
    """

    engine_from_req = ["RL", "NCF", "Random"]
    default_engine = ["RL", "NCF", "Random"]
    engines_keys = list(get_engines.keys())

    # 1. case - engine_from_req is specified, and engines_keys are the same as the range of engine_from_req
    for eg_req in engine_from_req:
        for eg_def in default_engine:
            engine_names = get_engine_names(eg_req, eg_def, engines_keys)
            expected_eg_names = [eg_req]

            if eg_def != eg_req:
                expected_eg_names.append(eg_def)
            list_diff = [eng for eng in engine_from_req if eng not in expected_eg_names]
            expected_eg_names.extend(list_diff)

            assert engine_names == expected_eg_names

    # 2. case - engine_from_req is NOT specified, and engines_keys are the same as the range of default_engine
    engine_from_req = None
    for eg_def in default_engine:
        engine_names = get_engine_names(engine_from_req, eg_def, engines_keys)
        expected_eg_names = [eg_def]
        list_diff = [eng for eng in default_engine if eng not in expected_eg_names]
        expected_eg_names.extend(list_diff)
        assert engine_names == expected_eg_names

    # 3. case - engine_from_req and default_engine are NOT specified
    engine_from_req = None
    default_engine = None
    engine_names = get_engine_names(engine_from_req, default_engine, engines_keys)
    expected_eg_names = [engines_keys[0], engines_keys[1], engines_keys[2]]
    assert engine_names == expected_eg_names

    # 4. case - engines_keys includes engines than engine_from_req nad default_engine
    engine_from_req = ("RL", "NCF", "Random")
    default_engine = ("RL", "NCF", "Random")
    # Last index of engines_keys, so it is expected to be the last element of returned names
    engines_keys.append("New Engine")

    for eg_req in engine_from_req:
        for eg_def in default_engine:
            engine_names = get_engine_names(eg_req, eg_def, engines_keys)
            expected_eg_names = [eg_req]
            if eg_def != eg_req:
                expected_eg_names.append(eg_def)
            list_diff = [eng for eng in engine_from_req if eng not in expected_eg_names]
            expected_eg_names.extend(list_diff)
            expected_eg_names.append("New Engine")

            assert engine_names == expected_eg_names


def test_engine_loader(
    get_engines,
    generate_users_and_services,
    mock_rl_pipeline_exec,
    mock_ncf_pipeline_exec,
):
    """
    Expected behaviour:
    return proper engine based on the order of engine names, else raise NoSavedMLComponentError
    """
    engine_names = ["RL", "NCF", "random"]
    engines = get_engines
    K = 3

    # 1. case RL and NCF engines are saved to the DB
    engine, engine_name = engine_loader(engine_names, engines, K)
    assert type(engine) == RLInferenceComponent
    assert engine_name == "RL"

    engine_names = ["NCF", "RL", "random"]
    engine, engine_name = engine_loader(engine_names, engines, K)
    assert type(engine) == NCFInferenceComponent
    assert engine_name == "NCF"

    engine_names = ["Random", "NCF", "RL"]
    engine, engine_name = engine_loader(engine_names, engines, K)
    assert type(engine) == RandomInferenceComponent
    assert engine_name == "Random"

    # 2. case no ML engine is saved and random engine is not passed
    engine_names = ["NCF", "RL"]
    MLComponent.drop_collection()
    with pytest.raises(NoSavedMLComponentError):
        engine_loader(engine_names, engines, K)

    engine_names = ["RL", "NCF"]
    with pytest.raises(NoSavedMLComponentError):
        engine_loader(engine_names, engines, K)

    # 3. case no ML engine is but random engine is also passed
    engine_names = ["NCF", "RL", "Random"]
    engine, engine_name = engine_loader(engine_names, engines, K)
    assert type(engine) == RandomInferenceComponent
    assert engine_name == "Random"


def test_engine_loader2(
    get_engines, generate_users_and_services, mock_rl_pipeline_exec
):
    # 4. case NCF engine is requested but does not exist - RL engine should be returned
    engine_names = ["NCF", "RL", "random"]
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
    # 5. case RL engine is requested but does not exist - NCF engine should be returned
    engine_names = ["RL", "NCF", "random"]
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
    Return proper engine and its name based on "engine_version"
    and whether user is logged-in or not
    """
    # 1. case user is logged-in
    recommendation_data["engine_version"] = "NCF"
    engine, engine_name = load_engine(recommendation_data)
    assert type(engine) == NCFInferenceComponent
    assert engine_name == "NCF"

    recommendation_data["engine_version"] = "RL"
    engine, engine_name = load_engine(recommendation_data)
    assert type(engine) == RLInferenceComponent
    assert engine_name == "RL"

    # 2. user is random
    del recommendation_data["user_id"]

    for engine_version in {"RL", "NCF", "Random", "random", "placeholder"}:
        recommendation_data["engine_version"] = engine_version
        engine, engine_name = load_engine(recommendation_data)
        assert type(engine) == RandomInferenceComponent
        assert engine_name == "Random"


def test_load_engine_with_aai_uid(
    generate_users_and_services,
    mock_rl_pipeline_exec,
    mock_ncf_pipeline_exec,
    recommendation_data_with_aai_uid,
):
    """
    Expected behaviour:
    Return proper engine and its name based on "engine_version"
    and whether user is logged-in or not
    """
    # 1. case user is logged-in
    recommendation_data_with_aai_uid["engine_version"] = "NCF"
    engine, engine_name = load_engine(recommendation_data_with_aai_uid)
    assert type(engine) == NCFInferenceComponent
    assert engine_name == "NCF"

    recommendation_data_with_aai_uid["engine_version"] = "RL"
    engine, engine_name = load_engine(recommendation_data_with_aai_uid)
    assert type(engine) == RLInferenceComponent
    assert engine_name == "RL"

    # 2. user is random
    del recommendation_data_with_aai_uid["aai_uid"]

    for engine_version in {"RL", "NCF", "random", "Random", "placeholder"}:
        recommendation_data_with_aai_uid["engine_version"] = engine_version
        engine, engine_name = load_engine(recommendation_data_with_aai_uid)
        assert type(engine) == RandomInferenceComponent
        assert engine_name == "Random"
