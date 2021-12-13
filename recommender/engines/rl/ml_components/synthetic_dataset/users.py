# pylint: disable=invalid-name, no-member, missing-module-docstring, unspecified-encoding

import json
import os
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from inflection import underscore, pluralize

from definitions import ROOT_DIR
from recommender.engines.autoencoders.training.data_preparation_step import (
    create_transformer,
    precalculate_tensors,
)
from recommender.engines.autoencoders.training.data_extraction_step import USERS
from recommender.models import User, Service, Category, ScientificDomain

SYNTHETIC_USERS = "synthetic_users"
CLUSTER_NAMES = [
    "humanities",
    "earth",
    "medicine",
    "biology",
    "physics",
    "computer_science",
    "business",
]
CATEGORY_CLUSTERS_PATH = os.path.join(ROOT_DIR, "resources", "category_clusters.json")
SCIENTIFIC_DOMAIN_CLUSTERS_PATH = os.path.join(
    ROOT_DIR, "resources", "scientific_domain_clusters.json"
)


def _sample_niche(
    cluster: str, clusters_json: Dict[str, List], sample_range: Tuple[int, int] = (0, 3)
) -> List[str]:
    """Sample either categories or scientific_domains
    that are associated with given cluster."""
    matching_niches = list(
        map(lambda x: x[0], filter(lambda x: cluster in x[1], clusters_json.items()))
    )
    n = min(
        np.random.randint(sample_range[0], sample_range[1] + 1), len(matching_niches)
    )
    return np.random.choice(matching_niches, n, replace=False).tolist()


def _filter_relevant(clazz) -> List[str]:
    """Returns either categories or scientific_domains that
    have more than 1 service associated with them"""
    assert clazz in [Category, ScientificDomain]
    histogram = {}

    for x in clazz.objects:
        count = len(
            Service.objects(**{f"{pluralize(underscore(clazz.__name__))}__in": [x]})
        )
        if histogram.get(x.name):
            histogram[x.name] += count
        else:
            histogram[x.name] = count

    histogram_df = pd.DataFrame(
        list(histogram.items()), columns=["class", "count"]
    ).set_index("class")
    return histogram_df[histogram_df["count"] >= 1].index.values.tolist()


def _synthesize_user(
    cluster: str,
    category_clusters: Dict[str, List],
    scientific_domain_clusters: Dict[str, List],
    niche_range: Tuple[int, int] = (5, 7),
) -> User:
    """Synthesizes a single user using the category/scientific_domain
    mapping to each cluster"""

    sampled_categories = _sample_niche(cluster, category_clusters, niche_range)
    sampled_scientific_domains = _sample_niche(
        cluster, scientific_domain_clusters, niche_range
    )

    user = User(
        id=max(User.objects.distinct("id") + [-1]) + 1,
        categories=Category.objects(name__in=sampled_categories),
        scientific_domains=ScientificDomain.objects(
            name__in=sampled_scientific_domains
        ),
        synthetic=True,
    ).save()

    return user


def synthesize_users(
    samples: int, cluster_distributions: Optional[Tuple[(float,) * 7]] = None
) -> List[User]:
    """
    Synthesizes any number of users according to distributions of the user clusters.
    Creates and saves an sklearn transformer for them.

    Args:
        samples: number of users to synthesize
        cluster_distributions: an array of length == len(CLUSTER_NAMES), defines
            the desired distribution of the user clusters. Must sum to 1.
    """
    with open(CATEGORY_CLUSTERS_PATH, encoding="utf-8") as f:
        category_clusters = json.load(f)

    with open(SCIENTIFIC_DOMAIN_CLUSTERS_PATH, encoding="utf-8") as f:
        scientific_domain_clusters = json.load(f)

    relevant_categories = _filter_relevant(Category)
    relevant_scientific_domains = _filter_relevant(ScientificDomain)

    category_clusters = {
        cat: cluster
        for cat, cluster in category_clusters.items()
        if cat in relevant_categories
    }
    scientific_domain_clusters = {
        sd: cluster
        for sd, cluster in scientific_domain_clusters.items()
        if sd in relevant_scientific_domains
    }

    users = []

    for _ in range(samples):
        cluster = np.random.choice(CLUSTER_NAMES, p=cluster_distributions)
        users.append(
            _synthesize_user(cluster, category_clusters, scientific_domain_clusters)
        )

    transformer = create_transformer(USERS)
    precalculate_tensors(users, transformer)

    return users
