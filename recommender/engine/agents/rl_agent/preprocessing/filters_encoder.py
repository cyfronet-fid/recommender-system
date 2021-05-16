# pylint: disable=no-member, too-few-public-methods

"""Implementation of the Filters Encoder"""
from typing import Optional

import torch
import torch.nn
from pandas import DataFrame
from sklearn.compose import ColumnTransformer

from recommender.engine.preprocessing.transformers import NoSavedTransformerError
from recommender.errors import MissingComponentError
from recommender.engine.models.autoencoders import SERVICES_AUTOENCODER, create_embedder
from recommender.engine.preprocessing.preprocessing import SERVICE_COLUMNS
from recommender.engine.utils import load_last_module, NoSavedModuleError
from recommender.engine.preprocessing import load_last_transformer, SERVICES
from recommender.models import (
    SearchData,
)


def _filters_to_df(search_data: SearchData) -> DataFrame:
    """Transform filters from search_data object into service-like Pandas dataframe."""

    dataframe = DataFrame(columns=SERVICE_COLUMNS)

    row_dict = {
        "countries": list(search_data.geographical_availabilities),
        "categories": [category.name for category in search_data.categories],
        "providers": [provider.name for provider in search_data.providers],
        "resource_organisation": [],
        "platforms": [platform.name for platform in search_data.related_platforms],
        "scientific_domains": [
            sci_dom.name for sci_dom in search_data.scientific_domains
        ],
        "target_users": [(tu.name, tu.description) for tu in search_data.target_users],
        "access_modes": [],
        "access_types": [],
        "trls": [],
        "life_cycle_statuses": [],
    }

    dataframe = dataframe.append(row_dict, ignore_index=True)

    return dataframe


class FiltersEncoder:
    """Encoder of filters that uses services transformer and embedder."""

    def __init__(
        self,
        service_transformer: Optional[ColumnTransformer] = None,
        service_embedder: Optional[torch.nn.Module] = None,
    ) -> None:
        """
        Args:
            service_transformer: Services transformer, by default most recent
             is loaded from the database.
            service_embedder: Services embedder, by default the encoder from
             the most recent Services Autoencoder is loaded from the database.
        """

        self.service_transformer = service_transformer
        self.service_embedder = service_embedder

        self._load_components()

    def __call__(self, search_data: SearchData) -> torch.Tensor:
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

        dataframe = _filters_to_df(search_data)

        with torch.no_grad():
            one_hot_array = self.service_transformer.transform(dataframe)
            one_hot_tensor = torch.Tensor(one_hot_array)
            dense_tensor = self._embed_filters(one_hot_tensor)

        return dense_tensor

    def _embed_filters(self, filters_oh_tensor: torch.Tensor) -> torch.Tensor:
        """
        Filters cannot be directly embedded as a single example batch because it
         is not handled by torch BatchNorm1D. To embedd it successfully it has
         to be multiplied and embedded after it. Finally only one embedded
         tensor is returned.

        There is a "safe_batch_size" constant that has been arbitrarily set to
         64 (there are some proofs that batchnorm can perform poorly if
         batch_size is small (<32). It is probably important during training
         - not inference - but this part of code will need some attention
         during polishing this project. It is possible that safe_batch_size
         should be same as batch_size during training. So, to sum it up, it's a
          TODO)

        Args:
            filters_oh_tensor: One-hot filters tensor.
            services_embedder: Service Embedder model.

        Returns:
            embedded_filters_tensor: Embedded user tensor
        """

        safe_batch_size = 64
        filters_tensors_batch = torch.cat([filters_oh_tensor] * safe_batch_size, dim=0)

        with torch.no_grad():
            embedded_filters_tensors_batch = self.service_embedder(
                filters_tensors_batch
            )
        embedded_filters_tensor = embedded_filters_tensors_batch[0].reshape(-1)

        return embedded_filters_tensor

    def _load_components(self):
        try:
            self.service_transformer = (
                self.service_transformer or load_last_transformer(SERVICES)
            )
        except NoSavedTransformerError as no_saved_transformer:
            raise MissingComponentError from no_saved_transformer

        try:
            self.service_embedder = self.service_embedder or create_embedder(
                load_last_module(SERVICES_AUTOENCODER)
            )
        except NoSavedModuleError as no_saved_module:
            raise MissingComponentError from no_saved_module

        self.service_embedder.eval()
