# pylint: disable=no-member, too-few-public-methods

"""Implementation of the Filters Embedder"""

from typing import Dict, Any

import torch
from pandas import DataFrame

from recommender.engine.pre_agent.models import load_last_module, SERVICES_AUTOENCODER
from recommender.engine.pre_agent.preprocessing import load_last_transformer, SERVICES
from recommender.engine.pre_agent.preprocessing.preprocessing import SERVICE_COLUMNS
from recommender.models import (
    Category,
    Provider,
    Platform,
    ScientificDomain,
    TargetUser,
)


def filters_to_df(search_data: Dict[str, Any]) -> DataFrame:
    """
    Transform filters from search_data dict into service-like Pandas dataframe.

    Args:
        search_data: Search data with filters.

    Returns:
        Service-like pandas dataframe.
    """

    dataframe = DataFrame(columns=SERVICE_COLUMNS)

    countries = search_data.get("geographical_availabilities")
    categories = Category.objects(id__in=search_data.get("categories"))
    providers = Provider.objects(id__in=search_data.get("providers"))
    platforms = Platform.objects(id__in=search_data.get("related_platforms"))
    scientific_domains = ScientificDomain.objects(
        id__in=search_data.get("scientific_domains")
    )
    target_users = TargetUser.objects(id__in=search_data.get("target_users"))

    row_dict = {
        "countries": list(countries),
        "categories": [category.name for category in categories],
        "providers": [provider.name for provider in providers],
        "resource_organisation": [],
        "platforms": [platform.name for platform in platforms],
        "scientific_domains": [sci_dom.name for sci_dom in scientific_domains],
        "target_users": [(tu.name, tu.description) for tu in target_users],
        "access_modes": [],
        "access_types": [],
        "trls": [],
        "life_cycle_statuses": [],
    }

    dataframe = dataframe.append(row_dict, ignore_index=True)

    return dataframe


class FiltersEmbedder:
    """Embedder of filters"""

    def __init__(self, services_transformer=None, service_embedder=None):
        """
        Args:
            services_transformer: Services transformer, by default most recent
             is loaded from the database.
            service_embedder: Services embedder, by default the encoder from
             the most recent Services Autoencoder is loaded from the database.
        """

        self.services_transformer = services_transformer or load_last_transformer(
            SERVICES
        )
        self.services_embedder = (
            service_embedder or load_last_module(SERVICES_AUTOENCODER).encoder
        )

    def __call__(self, search_data: Dict[str, Any]) -> torch.Tensor:
        """
        It prepares filters from search_data to look like service dataframe.
        This is how it maps data from search phrase:
            -> q:                           skip
            -> categories:                  get directly
            -> geographical_availabilities: get as "countries"
            -> order_type:                  skip
               TODO: get directly order_type, currently not available in any
                     preprocessor need to add it to places like: preprocessor,
                      factories, faker, etc.
            -> providers:                   get directly
            -> related_platforms:           get as "platforms"
            -> scientific_domains:          get directly
            -> sort:                        skip
            -> target_users:                get directly

        Next, it embed filters like a service using services transformer and
         services embedder

        Args:
            search_data: Search data with filters.

        Returns:
            Tensor of the embedded filters.
        """

        dataframe = filters_to_df(search_data)

        with torch.no_grad():
            one_hot_array = self.services_transformer.transform(dataframe)
            one_hot_tensor = torch.Tensor(one_hot_array)
            dense_tensor = self.services_embedder(one_hot_tensor)
            dense_tensor = dense_tensor.reshape(-1)

        return dense_tensor
