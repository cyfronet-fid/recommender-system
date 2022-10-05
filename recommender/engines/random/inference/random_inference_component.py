# pylint: disable=invalid-name, too-few-public-methods, line-too-long
"""Recommender engine that provides users with random recommendations in a given context"""

import random
from typing import List, Tuple, Dict, Any

from recommender.engines.base.base_inference_component import BaseInferenceComponent
from recommender.engines.explanations import Explanation
from recommender.services.fts import retrieve_services_for_recommendation


class RandomInferenceComponent(BaseInferenceComponent):
    """
    Recommender engine that provides all users with random recommendations in a given context.

    Used for:
        - random users,
        - all users if necessary (for example during ML training).
    """

    engine_name = "random"
    default_explanation = Explanation(
        long="This service has been selected at random however taking into"
        " account the search criteria",
        short="This service has been selected randomly.",
    )

    def __call__(
        self, context: Dict[str, Any]
    ) -> Tuple[List[int], List[float], List[Explanation]]:
        """
        Get random recommendations in a given context.

        Args:
            context: json dict from the /recommendations endpoint request.

        Returns:
            Tuple of recommended services ids.
        """
        candidates, _ = self._get_recommendation_context(context)

        return self._generate_recommendations(candidates)

    def _generate_recommendations(
        self, candidates: Tuple[int]
    ) -> Tuple[List[int], List[float], List[Explanation]]:
        """
        Generate recommendations.

        Args:
            candidates: item space from the Marketplace.

        Returns:
            recommended_services_ids: List of recommended services ids.
        """
        candidate_services = list(retrieve_services_for_recommendation(candidates))

        recommended_services = random.sample(list(candidate_services), self.K)
        recommended_services_ids = [s.id for s in recommended_services]

        scores = self.K * [
            1 / len(candidate_services)
        ]  # Services have been sampled uniformly
        explanations = self._generate_explanations()

        return recommended_services_ids, scores, explanations
