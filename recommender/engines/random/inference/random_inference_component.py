# pylint: disable=invalid-name, too-few-public-methods, line-too-long
"""Recommender engine that provides users with random recommendations in a given context"""

import random
from typing import List, Tuple, Dict, Any

from recommender.engines.base.base_inference_component import BaseInferenceComponent
from recommender.services.fts import retrieve_services_for_recommendation


class RandomInferenceComponent(BaseInferenceComponent):
    """
    Recommender engine that provides all users with random recommendations in a given context.

    Used for:
        - random users,
        - all users if necessary (for example during ML training).
    """

    engine_name = "random"

    def __call__(self, context: Dict[str, Any]) -> List[int]:
        """
        Get random recommendations in a given context.

        Args:
            context: json dict from the /recommendations endpoint request.

        Returns:
            Tuple of recommended services ids.
        """
        elastic_services, _ = self._get_recommendation_context(context)

        return self._generate_recommendations(elastic_services)

    def _generate_recommendations(self, elastic_services: Tuple[int]) -> List[int]:
        """
        Generate recommendations.

        Args:
            elastic_services: item space from the Marketplace.

        Returns:
            recommended_services_ids: List of recommended services ids.
        """
        candidate_services = list(
            retrieve_services_for_recommendation(elastic_services)
        )

        recommended_services = random.sample(list(candidate_services), self.K)
        recommended_services_ids = [s.id for s in recommended_services]

        return recommended_services_ids
