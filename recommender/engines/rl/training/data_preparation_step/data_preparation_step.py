# pylint: disable=fixme, missing-module-docstring, missing-class-docstring, invalid-name

import time
from typing import Tuple

from torch.utils.data import DataLoader

from recommender.engines.autoencoders.ml_components.embedder import Embedder
from recommender.engines.autoencoders.ml_components.normalizer import Normalizer
from recommender.engines.rl.ml_components.sars_encoder import SarsEncoder
from recommender.engines.base.base_steps import DataPreparationStep
from recommender.engines.rl.training.data_preparation_step.replay_buffer_v2 import (
    ReplayBufferV2,
)
from recommender.models import User, Service


class RLDataPreparationStep(DataPreparationStep):
    def __init__(self, pipeline_config):
        super().__init__(pipeline_config)
        self.batch_size = self.resolve_constant("batch_size", 64)
        self.shuffle = self.resolve_constant("shuffle", True)
        self.user_embedder = Embedder.load(version="user")
        self.service_embedder = Embedder.load(version="service")
        self.sars_encoder = SarsEncoder(
            self.user_embedder,
            self.service_embedder,
            use_cached_embeddings=True,
            save_cached_embeddings=False,
        )

    def _cache_and_normalize(self, data):
        # TODO: we should be doing this before the pipeline runs, same with NCF pipeline
        self.user_embedder(User.objects, use_cache=False, save_cache=True)
        self.service_embedder(Service.objects, use_cache=False, save_cache=True)
        normalizer = Normalizer()
        normalizer(User.objects, save_cache=True)
        normalizer(Service.objects, save_cache=True)

        for x in data:
            x.reload()

    def __call__(self, data=None) -> Tuple[DataLoader, dict]:
        encoding_start = time.time()
        self._cache_and_normalize(data)

        encoded_sarses = self.sars_encoder(data)
        encoding_end = time.time()

        replay_buffer = ReplayBufferV2(encoded_sarses)
        training_dl = DataLoader(
            replay_buffer, batch_size=self.batch_size, shuffle=self.shuffle
        )

        return training_dl, {
            "encoding_time": encoding_end - encoding_start,
            "no_of_batches": len(training_dl),
        }
