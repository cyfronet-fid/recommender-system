# pylint: disable=missing-module-docstring, missing-class-docstring

from time import time
from typing import Tuple, List

from recommender.engines.autoencoders.ml_components.embedder import Embedder
from recommender.engines.base.base_steps import DataExtractionStep
from recommender.models import Sars
from recommender.services.sarses_generator import generate_sarses
from recommender.services.synthetic_dataset.dataset import generate_synthetic_sarses
from recommender.services.synthetic_dataset.rewards import RewardGeneration


class RLDataExtractionStep(DataExtractionStep):
    def __init__(self, pipeline_config):
        super().__init__(pipeline_config)
        self.generate_new = self.resolve_constant("generate_new", True)
        self.synthetic = self.resolve_constant("synthetic", True)
        self.synthetic_params = self.resolve_constant(
            "synthetic_params",
            {
                "interactions_range": (3, 5),
                "reward_generation_mode": RewardGeneration.COMPLEX,
            },
        )

    def __call__(self, data=None) -> Tuple[List[Sars], dict]:
        start = time()
        sarses = self._generate_proper_sarses()
        end = time()
        return sarses, {"sarses_generation_duration": end - start}

    def _generate_proper_sarses(self):
        if self.generate_new:
            Sars.objects(synthetic=self.synthetic).delete()

            if self.synthetic:
                service_embedder = Embedder.load(version="service")
                return generate_synthetic_sarses(
                    service_embedder, **self.synthetic_params
                )

            return list(generate_sarses())

        return list(Sars.objects(synthetic=self.synthetic))
