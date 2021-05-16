# pylint: disable=no-else-return, no-member, missing-class-docstring

"""Functions related to Scikit-learn transformers used in other functions
 of data preprocessing pipeline
"""
import pickle
from typing import Optional

from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

from recommender.engine.preprocessing.common import USERS, SERVICES
from recommender.engine.preprocessing.list_column_one_hot_encoder import (
    ListColumnOneHotEncoder,
)
from recommender.models import ScikitLearnTransformer


def create_users_transformer():
    """Creates users transformer"""

    transformer = make_column_transformer(
        (make_pipeline(ListColumnOneHotEncoder()), ["scientific_domains", "categories"])
    )

    return transformer


def create_services_transformer():
    """Creates services transformer"""

    transformer = make_column_transformer(
        (
            make_pipeline(ListColumnOneHotEncoder()),
            [
                "countries",
                "categories",
                "providers",
                "resource_organisation",
                "scientific_domains",
                "platforms",
                "target_users",
                "access_modes",
                "access_types",
                "trls",
                "life_cycle_statuses",
            ],
        )
    )

    return transformer


def create_transformer(name):
    """Creates new transformer of given name."""

    if name == USERS:
        return create_users_transformer()
    elif name == SERVICES:
        return create_services_transformer()

    raise ValueError


def save_transformer(
    transformer, name: Optional[str] = None, description: Optional[str] = None
):
    """It saves transformer to database using pickle"""

    ScikitLearnTransformer(
        name=name, description=description, binary_transformer=pickle.dumps(transformer)
    ).save()


class NoSavedTransformerError(Exception):
    pass


def load_last_transformer(name):
    """It loads transformer from database and unpickles it"""

    last_transformer_model = (
        ScikitLearnTransformer.objects(name=name).order_by("-id").first()
    )

    if last_transformer_model is None:
        raise NoSavedTransformerError(f"No saved transformer with name {name}!")

    transformer = pickle.loads(last_transformer_model.binary_transformer)

    return transformer
