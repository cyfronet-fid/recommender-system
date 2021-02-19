# pylint: disable=no-else-return, no-member

"""Functions related to Scikit-learn transformers used in other functions
 of data preprocessing pipeline
"""

import pickle

from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

from recommender.engine.pre_agent.preprocessing.mongo_to_dataframe import (
    USERS,
    SERVICES,
    LABELS,
)
from recommender.engine.pre_agent.utilities.list_column_one_hot_encoder import (
    ListColumnOneHotEncoder,
)
from recommender.engine.pre_agent.utilities.pipeline_friendly_label_binarizer import (
    PipelineFriendlyLabelBinarizer,
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


def create_labels_transformer():
    """Creates labels transformer"""

    transformer = make_column_transformer(
        (
            make_pipeline(PipelineFriendlyLabelBinarizer()),
            ["ordered"],
        )
    )

    return transformer


def create_transformer(name):
    """Creates new transformer of given name."""

    if name == USERS:
        return create_users_transformer()
    elif name == SERVICES:
        return create_services_transformer()
    elif name == LABELS:
        return create_labels_transformer()

    raise ValueError


def create_transformers():
    """Creates users, services and labels transformers"""

    user_transformer = create_users_transformer()
    service_transformer = create_services_transformer()
    label_transformer = create_labels_transformer()

    transformers = {
        USERS: user_transformer,
        SERVICES: service_transformer,
        LABELS: label_transformer,
    }

    return transformers


def save_transformer(transformer, name=None, description=None):
    """It saves transformer to database using pickle"""

    ScikitLearnTransformer(
        name=name, description=description, binary_transformer=pickle.dumps(transformer)
    ).save()


def load_last_transformer(name):
    """It loads transformer from database and unpickles it"""

    transformer = pickle.loads(
        ScikitLearnTransformer.objects(name=name)
        .order_by("-id")
        .first()
        .binary_transformer
    )

    return transformer
