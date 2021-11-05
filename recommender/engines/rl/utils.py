# pylint: disable=no-member, invalid-name, missing-function-docstring

"""RL Agent Utilities"""

from typing import Iterable, Union, List

import pandas as pd
from mongoengine import QuerySet


def create_index_id_map(services: Union[Iterable, QuerySet]) -> pd.DataFrame:
    return pd.DataFrame([s.id for s in services], columns=["id"])


def get_service_indices(index_id_map: pd.DataFrame, ids: List[int]) -> List[int]:
    """Given a mapping between indices in the embedding and
    database ids returns indices of services with given ids."""

    id_index_map = pd.Series(index_id_map["id"].index.values, index=index_id_map["id"])

    # Below intersection is necessary to avoid key error in pandas .loc[]
    # Checking for existence in set is O(1) because it's hash-based,
    # so the overall complexity is O(len(ids))
    possible_values = set(id_index_map.index.values.tolist())
    valid_ids = [x for x in ids if x in possible_values]

    indices = id_index_map.loc[valid_ids].values.reshape(-1).tolist()

    return indices

