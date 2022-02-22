# pylint: disable=invalid-name, no-member, missing-function-docstring
# pylint: disable=global-variable-undefined, too-many-branches, fixme

"""Project Utilities"""

import json
import random
import functools
from time import time
from datetime import datetime
from typing import Dict, List, Union, Optional, Any
from uuid import UUID

import graphviz
from bson import SON, ObjectId
from mongoengine import Document
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from definitions import ROOT_DIR, RUN_DIR
from recommender.engines.panel_id_to_services_number_mapping import PANEL_ID_TO_K
from recommender.engines.rl.utils import _get_visit_ids
from recommender.models import User, Service, UserAction, Recommendation
from recommender.services.fts import AVAILABLE_FOR_RECOMMENDATION
from logger_config import get_logger

logger = get_logger(__name__)


def _son_to_dict(son_obj: SON) -> dict:
    """
    Transform SON object into python dict.

    Args:
        son_obj: SON object

    Returns:
        dictionary: python dict
    """
    dictionary = dict(son_obj)
    for key, value in dictionary.items():
        if isinstance(value, SON):
            dictionary[key] = _son_to_dict(value)
        if isinstance(value, (ObjectId, UUID, datetime)):
            dictionary[key] = str(value)
    if "_cls" in dictionary.keys():
        dictionary.pop("_cls")
    return dictionary


def printable(obj: Document) -> str:
    """
    Make MongoEngine object pretty printable using `print` function :)

    Args:
        obj: MongoEngine object.

    Return:
        s: pretty formatted json dict string.
    """

    son_obj = obj.to_mongo()
    dictionary = _son_to_dict(son_obj)
    string = json.dumps(dictionary, indent=2)
    return string


def _get_services_with_non_empty_list_fileds():
    size_not_zero = {"$not": {"$size": 0}}
    q = Service.objects(
        __raw__={
            "categories": size_not_zero,
            "countries": size_not_zero,
            "providers": size_not_zero,
            "platforms": size_not_zero,
            "scientific_domains": size_not_zero,
            "target_users": size_not_zero,
        }
    )
    return q


def _get_search_data_examples(
    k: Optional[int] = None, deterministic: Optional[bool] = False
) -> Dict[str, List[Union[int, str]]]:
    """
    Generates examples of search_data fields based on Services in the database.

    Returns:
        examples: examples for each field of search_data
         (except: q, order_type, rating, sort)
    """

    # If some list fields of service are empty then this service won't be
    # found later because it will not match set of values (that will be
    # most probably not empty because of other services), so we have to use
    # this function:
    q = _get_services_with_non_empty_list_fileds()

    q = q(status__in=AVAILABLE_FOR_RECOMMENDATION)
    services = list(q)

    if k is None:
        k = 3

    if deterministic:
        services = services[:k]
    else:
        services = random.sample(services, k=k)

    categories_ids = set()
    geographical_availabilities = set()
    provider_ids = set()
    related_platform_ids = set()
    scientific_domain_ids = set()
    target_user_ids = set()
    for service in services:
        categories_ids.update([c.id for c in service.categories])
        geographical_availabilities.update(service.countries)
        provider_ids.update([p.id for p in service.providers])
        related_platform_ids.update([rp.id for rp in service.platforms])
        scientific_domain_ids.update([sd.id for sd in service.scientific_domains])
        target_user_ids.update([tu.id for tu in service.target_users])

    examples = {
        "categories": list(categories_ids),
        "geographical_availabilities": list(geographical_availabilities),
        "providers": list(provider_ids),
        "related_platforms": list(related_platform_ids),
        "scientific_domains": list(scientific_domain_ids),
        "target_users": list(target_user_ids),
    }

    return examples


def gen_json_dict(panel_id: str, anonymous_user: bool = False) -> Dict[str, Any]:
    """
    Generate json_dict ready for using in any agent based on database
     and provided panel_id.

    Args:
        panel_id: Version of the panel, could be `"v1"` or `"v1"`

    Returns:
        json_dict: dictionary compatible with body of the /recommendations
         endpoint.
    """

    K = PANEL_ID_TO_K.get(panel_id)

    search_data = _get_search_data_examples(k=K, deterministic=True)
    search_data["q"] = ""

    json_dict = {
        "unique_id": "5642c351-80fe-44cf-b606-304f2f338122",
        "timestamp": "2021-05-21T18:43:12.295Z",
        "visit_id": "202090a4-de4c-4230-acba-6e2931d9e37c",
        "page_id": "/services",
        "panel_id": panel_id,
        "search_data": search_data,
    }

    if not anonymous_user:
        json_dict["user_id"] = User.objects.first().id

    return json_dict


def load_examples() -> Dict:
    """
    If possible, generates examples for /recommendations request using
     appropriate database.
    If not, generates artificial examples.

    Returns:
        examples: Examples dict.
    """

    examples = {
        "categories": [1, 2, 3],
        "geographical_availabilities": ["WW"],
        "providers": [1, 2, 3],
        "related_platforms": [1, 2, 3],
        "scientific_domains": [1, 2, 3],
        "target_users": [1, 2, 3],
    }

    # TODO: Commented, as it yet again breaks the application xd - migrations this time
    # Below, `os.environ["FLASK_ENV"]` and manual connection is used rather
    # than standard flask DB connection, because this code is executed before
    # flask app building is finished. It has to be done in this way to provide
    # realistic /recommendations endpoint examples in the swagger.

    # if os.environ.get("FLASK_ENV") == "testing":
    #     return examples
    #
    # if os.environ.get("FLASK_ENV") == "development":
    #     host = DevelopmentConfig.MONGODB_HOST
    # elif os.environ.get("FLASK_ENV") == "production":
    #     host = ProductionConfig.MONGODB_HOST
    # else:
    #     return examples
    # connect(host=host)
    # examples = _get_search_data_examples(
    #     k=max(list(PANEL_ID_TO_K.values())), deterministic=True
    # )
    # disconnect()

    return examples


def timeit(func):
    if "performance_measurements" not in globals():
        global performance_measurements
        performance_measurements = {}

    if "writer" not in globals():
        global writer
        writer = SummaryWriter(log_dir=RUN_DIR)

    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        start = time()
        ret_val = func(*args, **kwargs)
        end = time()
        elapsed = end - start

        key = f"{func.__name__}"
        if isinstance(performance_measurements.get(key), list):
            performance_measurements[key].append(elapsed)
        elif performance_measurements.get(key) is None:
            performance_measurements[key] = [elapsed]

        step = len(performance_measurements[key])
        writer.add_scalars("Performance", {key: elapsed}, step)
        writer.flush()

        return ret_val

    return newfunc


def show_times():
    if "performance_measurements" not in globals():
        global performance_measurements
        performance_measurements = {}

    for key, value in performance_measurements.items():
        records = len(value)
        mean = sum(value) / records
        logger.info(
            "[%s] Mean execution time: %s, records number: %s", key, mean, records
        )

    return performance_measurements


def clear_times():
    if "performance_measurements" not in globals():
        global performance_measurements
    performance_measurements = {}


def visualize_uas(filename=None, view=True, save=False):
    """Visualize all user actions as a graph in the svg file"""

    if save:
        filename = filename or ROOT_DIR / "graphs" / (
            "user_actions_visualization" + "_" + str(time())
        )
    else:
        filename = None
    graph = graphviz.Digraph(name="Container")

    for user_action in tqdm(UserAction.objects.order_by("+timestamp")):
        ua_svid, ua_tvid = _get_visit_ids(user_action)

        if user_action.action.order:
            action_color = "red"
            action_label = "Order"
        else:
            action_color = "black"
            action_label = ""

        # Source
        source_label = ua_svid[:4]
        source_color = "black"

        recommendation = Recommendation.objects(visit_id=ua_svid).first()
        if recommendation is not None:
            if (
                user_action.source.root is not None
                and user_action.source.root.service is not None
            ):
                action_color = "green"
                action_label = f"via service(id={user_action.source.root.service.id})"

            source_color = "green"
            source_label = f"{source_label}\n(recommendation)"

        graph.node(ua_svid, label=source_label, color=source_color)

        # Target
        target_label = ua_tvid[:4]
        target_color = "black"
        recommendation = Recommendation.objects(visit_id=ua_tvid).first()
        if recommendation is not None:
            target_color = "green"
            target_label = f"{target_label}\n(recommendation)"

        graph.node(ua_tvid, label=target_label, color=target_color)

        graph.edge(ua_svid, ua_tvid, color=action_color, label=action_label)

    logger.info("Rendering graph into SVG file...")
    graph.render(filename=filename, format="svg", view=view)
