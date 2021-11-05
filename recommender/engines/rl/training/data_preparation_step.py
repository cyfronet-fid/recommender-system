# pylint: disable=fixme, missing-module-docstring, missing-class-docstring, invalid-name

import time
from typing import Tuple

from torch.utils.data import DataLoader

from recommender.engines.autoencoders.ml_components.embedder import (
    Embedder,
    USER_EMBEDDER,
    SERVICE_EMBEDDER,
)
from recommender.engines.rl.ml_components.encoders.sars_encoder import (
    SarsEncoder,
)
from recommender.engines.base.base_steps import DataPreparationStep
from recommender.engines.rl.ml_components.replay_buffer import ReplayBuffer

SARS_BATCH_SIZE = "sars_batch_size"
SHUFFLE = "shuffle"


class RLDataPreparationStep(DataPreparationStep):
    def __init__(self, pipeline_config):
        super().__init__(pipeline_config)
        self.batch_size = self.resolve_constant(SARS_BATCH_SIZE, 64)
        self.shuffle = self.resolve_constant(SHUFFLE, True)
        self.user_embedder = Embedder.load(version=USER_EMBEDDER)
        self.service_embedder = Embedder.load(version=SERVICE_EMBEDDER)
        self.sars_encoder = SarsEncoder(self.user_embedder, self.service_embedder)

    def __call__(self, data=None) -> Tuple[tuple, dict]:
        encoding_start = time.time()
        sarses = data
        encoded_sarses = self.sars_encoder(sarses)
        encoding_end = time.time()

        replay_buffer = ReplayBuffer(encoded_sarses)
        replay_buffer_dl = DataLoader(
            replay_buffer, batch_size=self.batch_size, shuffle=self.shuffle
        )

        return (replay_buffer_dl, sarses), {
            "encoding_time": encoding_end - encoding_start,
            "no_of_batches": len(replay_buffer_dl),
        }
