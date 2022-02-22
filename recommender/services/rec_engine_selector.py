# pylint: disable=line-too-long, invalid-name

"""Select proper recommendation engine"""
import os
from typing import Dict, Any, List, Tuple

from recommender.engines.base.base_inference_component import BaseInferenceComponent
from recommender.engines.ncf.inference.ncf_inference_component import (
    NCFInferenceComponent,
)
from recommender.engines.rl.inference.rl_inference_component import RLInferenceComponent
from recommender.engines.panel_id_to_services_number_mapping import PANEL_ID_TO_K
from recommender.errors import (
    InvalidRecommendationPanelIDError,
    NoSavedMLComponentError,
)

NCF_ENGINE_NAME = "NCF"
RL_ENGINE_NAME = "RL"


def get_K(context: Dict[str, Any]) -> int:
    """
    Get the K constant from the context.

    Args:
        context: context json  from the /recommendations endpoint request.

    Returns:
        K constant.
    """
    K = PANEL_ID_TO_K.get(context.get("panel_id"))
    if K is None:
        raise InvalidRecommendationPanelIDError()
    return K


def get_default_recommendation_alg(
    env_variable: str = "DEFAULT_RECOMMENDATION_ALG",
) -> str:
    """
    Get the default recommendation algorithm from the .env

    Args:
        env_variable: the name of the default recommendation algorithm from .env
    """
    recommendation_alg = os.environ.get(env_variable, "RL").upper()

    if recommendation_alg == "NCF":
        return "NCF"
    return "RL"


def get_engine_names(
    engine_from_req: str,
    default_engine: str,
    engines_keys: List[str],
) -> List[str]:
    """
    Get engine names

    Order in which engine is selected:
    1) Engine from a body request if this name is in engines_keys,
    2) Default engine if this name is in engines_keys,
    3) Any engine name that exists.

    Args:
        engine_from_req - engine name requested in a body of the request
        default_engine - default recommendation engine name
        engines_keys - any engine name potentially available
    """
    engine_names = []

    for engine in (engine_from_req, default_engine):
        if engine in engines_keys and engine not in engine_names:
            engine_names.append(engine)

    for engine in engines_keys:
        if engine not in engine_names:
            engine_names.append(engine)

    return engine_names


def engine_loader(
    engine_names: List[str],
    engines: Dict[str, Any],
    K: int,
) -> Tuple[BaseInferenceComponent, str]:
    """
    Try loading engines in the right order to maximize the availability of recommendations

    Args:
        engine_names - all engine names
        engines - all available engines,
        K - number of requested recommendations
    """
    engine = None
    eg_name = None

    while not engine:
        for engine_name in engine_names:
            try:
                engine = engines.get(engine_name)(K)
                if engine:
                    eg_name = engine_name
                    break
            except NoSavedMLComponentError:
                pass

        if not engine:
            raise NoSavedMLComponentError()

    return engine, eg_name


def load_engine(json_dict: dict) -> Tuple[BaseInferenceComponent, str]:
    """
    Load the engine based on 'engine_version' parameter from a query

    Args:
        json_dict: A body from Marketplace's query

    Returns:
        engine: An instance of NCFInferenceComponent or RLInferenceComponent
    """
    engine_from_req = json_dict.get("engine_version")
    default_engine = get_default_recommendation_alg()
    K = get_K(json_dict)

    engines = {
        NCF_ENGINE_NAME: NCFInferenceComponent,
        RL_ENGINE_NAME: RLInferenceComponent,
    }

    engine_names = get_engine_names(
        engine_from_req, default_engine, list(engines.keys())
    )
    engine, engine_name = engine_loader(engine_names, engines, K)

    return engine, engine_name
