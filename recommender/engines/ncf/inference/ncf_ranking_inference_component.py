# pylint: disable=unused-argument, too-few-public-methods, missing-class-docstring, missing-function-docstring
# pylint: disable=superfluous-parens, no-else-return, no-member, not-callable, line-too-long

"""Neural Collaborative Filtering Ranking Inference Component"""

from typing import List, Tuple

from recommender.engines.explanations import Explanation
from recommender.engines.ncf.inference.ncf_inference_component import (
    NCFInferenceComponent,
)
from recommender.models import User, SearchData
from logger_config import get_logger

logger = get_logger(__name__)


class NCFRankingInferenceComponent(NCFInferenceComponent):
    """
    Recommender engine that provides logged-in users with personalized recommendations in a given context. It ranks all services.
    """

    engine_name = "NCFRanking"

    def _generate_recommendations(
        self, user: User, candidates: Tuple[int], search_data: SearchData
    ) -> Tuple[List[int], List[float], List[Explanation]]:
        """Generate services ranking for logged user.

        Args:
            user: user for whom recommendation will be generated.
            candidates: item space from the Marketplace.
            search_data: search phrase and filters information for narrowing
             down an item space.

        Returns:
            recommended_services_ids: List of recommended services ids.
            scores: List of ranking scores for all recommended services.
            explanations: List of explanations for all recommended services.
        """

        all_recommended_services_ids, all_scores = self._get_ranking(
            user, candidates, search_data
        )
        self.K = len(all_recommended_services_ids)
        explanations = self._generate_explanations()

        return all_recommended_services_ids, all_scores, explanations
