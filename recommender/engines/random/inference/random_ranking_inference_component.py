# pylint: disable=invalid-name, too-few-public-methods, line-too-long
"""Recommender engine that provides users with random ranking recommendations from a given context"""
import random
from recommender.engines.explanations import Explanation
from recommender.services.fts import retrieve_services_for_recommendation
from recommender.engines.random.inference.random_inference_component import (
    RandomInferenceComponent,
)


class RandomRankingInferenceComponent(RandomInferenceComponent):
    """Recommender engine that provides all users with random recommendations from a given context. It ranks all services"""

    engine_name = "RandomRanking"

    def _generate_recommendations(
        self, candidates: tuple[int]
    ) -> tuple[list[int], list[float], list[Explanation]]:
        """
        Sort recommendations randomly.

        Args:
            candidates: recommendations item space

        Returns:
            recommended_services_ids: List of sorted services
        """
        candidate_services = list(retrieve_services_for_recommendation(candidates))
        random.shuffle(candidate_services)

        recommended_services_ids = [s.id for s in candidate_services]

        scores = len(candidate_services) * [
            round(1 / len(candidate_services), 3)
        ]  # Services have been ranked uniformly

        self.K = len(candidate_services)
        explanations = self._generate_explanations()

        return recommended_services_ids, scores, explanations
