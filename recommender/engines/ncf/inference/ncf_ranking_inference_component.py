# pylint: disable=unused-argument, too-few-public-methods, missing-class-docstring, missing-function-docstring
# pylint: disable=no-self-use, superfluous-parens, no-else-return, no-member, not-callable, line-too-long

"""Neural Collaborative Filtering Ranking Inference Component"""

from typing import List, Tuple

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
        self, user: User, elastic_services: Tuple[int], search_data: SearchData
    ) -> List[int]:
        """Generate services ranking for logged user.

        Args:
            user: user for whom recommendation will be generated.
            elastic_services: item space from the Marketplace.
            search_data: search phrase and filters information for narrowing
             down an item space.

        Returns:
            services_ids_ranking: Ranked list of services IDs.
        """

        ranking = self._get_ranking(user, elastic_services, search_data)
        services_ids_ranking = [pair[1] for pair in ranking]

        return services_ids_ranking
