# pylint: disable=line-too-long, invalid-name

"""Select proper recommendation engine"""
import os
from typing import Dict, Any, List, Tuple

from recommender.engines.engines import ENGINES
from recommender.engines.random.inference.random_inference_component import (
    RandomInferenceComponent,
)
from recommender.engines.panel_id_to_services_number_mapping import PANEL_ID_TO_K
from recommender.errors import (
    InvalidRecommendationPanelIDError,
    NoSavedMLComponentError,
)


def get_K(context: Dict[str, Any]) -> int:
    """
    Get the K constant from the context.

    Args:
        context: context json  from the /recommendations endpoint request.

    Returns:
        K: constant which specifies how many recommendations are requested.
    """
    K = PANEL_ID_TO_K.get(context.get("panel_id"))
    if K is None:
        raise InvalidRecommendationPanelIDError()
    return K


def get_default_recommendation_alg(
    engine_names: Tuple[str],
    default_engine: str = "DEFAULT_RECOMMENDATION_ALG",
) -> str:
    """
    Get the default recommendation algorithm from the .env

    Args:
        engine_names: list of available engines'
        default_engine: the name of the default recommendation algorithm from .env

    Returns:
        rec_alg: default recommendation algorithm
    """
    rec_alg = os.environ.get(default_engine, "RL")
    rec_alg = rec_alg.upper() if len(rec_alg) in {2, 3} else rec_alg.lower()

    return rec_alg if rec_alg in engine_names else "RL"


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
        engine_from_req: engine name requested in a body of the request
        default_engine: default recommendation engine name
        engines_keys: any engine name potentially available

    Returns:
        engine_names: engines names to load in a proper order
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
) -> Tuple[Any, str]:
    """
    Try loading engines in the right order to maximize the availability of recommendations

    Args:
        engine_names: all engine names
        engines: all available engines,
        K: number of requested recommendations

    Returns:
        engine: engine for serving recommendations
        engine_name: engine name
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


def load_engine(json_dict: dict) -> Tuple[Any, str]:
    """
    Load the engine based on whether a user is logged in
     and 'engine_version' parameter from the query.

    Args:
        json_dict: A body from Marketplace's query

    Returns:
        engine: engine for serving recommendations
        engine_name: engine name
    """
    K = get_K(json_dict)

    if not json_dict.get("user_id"):  # user is anonymous
        return (
            RandomInferenceComponent(K),
            RandomInferenceComponent.engine_name,
        )

    engine_from_req = json_dict.get("engine_version")
    default_engine = get_default_recommendation_alg(tuple(ENGINES.keys()))

    engine_names = get_engine_names(
        engine_from_req, default_engine, list(ENGINES.keys())
    )
    engine, engine_name = engine_loader(engine_names, ENGINES, K)

    return engine, engine_name
