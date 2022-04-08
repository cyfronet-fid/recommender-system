# pylint: disable=too-few-public-methods

"""Embedding Component"""

from recommender.engines.autoencoders.ml_components.embedder import (
    Embedder,
    USER_EMBEDDER,
    SERVICE_EMBEDDER,
)
from recommender.engines.autoencoders.ml_components.normalizer import Normalizer
from recommender.models import User, Service
from logger_config import get_logger

logger = get_logger(__name__)


class EmbeddingComponent:
    """Embedding Component"""

    def __init__(self):
        self.user_embedder = Embedder.load(version=USER_EMBEDDER)
        self.services_embedder = Embedder.load(version=SERVICE_EMBEDDER)
        self.normalizer = Normalizer()

    def __call__(self, use_cache=False, save_cache=True, verbose=False):
        """Performs embedding with normalization."""

        if verbose:
            logger.info("Start embedding...")
            logger.info("Start converting one_hot_tensors into dense_tensors...")

        self.user_embedder(
            User.objects,
            use_cache=use_cache,
            save_cache=save_cache,
            version="Users",
            verbose=verbose,
        )
        self.services_embedder(
            Service.objects,
            use_cache=use_cache,
            save_cache=save_cache,
            version="Services",
            verbose=verbose,
        )

        if verbose:
            logger.info("Finished converting one_hot_tensors into dense_tensors!")
            logger.info("Start normalizing dense_tensors...")

        self.normalizer(
            User.objects, save_cache=save_cache, version="Users", verbose=verbose
        )
        self.normalizer(
            Service.objects, save_cache=save_cache, version="Services", verbose=verbose
        )

        if verbose:
            logger.info("Finished normalizing dense_tensors!")
            logger.info("Finished embedding!")
