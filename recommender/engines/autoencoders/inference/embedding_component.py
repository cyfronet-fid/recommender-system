# pylint: disable=missing-class-docstring, missing-class-docstring, too-few-public-methods, missing-module-docstring

from recommender.engines.autoencoders.ml_components.embedder import (
    Embedder,
    USER_EMBEDDER,
    SERVICE_EMBEDDER,
)
from recommender.engines.autoencoders.ml_components.normalizer import Normalizer
from recommender.models import User, Service


class EmbeddingComponent:
    def __init__(self):
        self.services_embedder = Embedder.load(version=SERVICE_EMBEDDER)
        self.user_embedder = Embedder.load(version=USER_EMBEDDER)
        self.normalizer = Normalizer()

    def __call__(self, use_cache=False, save_cache=True):
        print("Starting embedding...")
        self.user_embedder(User.objects, use_cache=use_cache, save_cache=save_cache)
        self.services_embedder(
            Service.objects, use_cache=use_cache, save_cache=save_cache
        )
        self.normalizer(User.objects, save_cache=save_cache)
        self.normalizer(Service.objects, save_cache=save_cache)
        print("Finished embedding!")
