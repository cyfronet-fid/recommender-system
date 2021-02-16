# pylint: disable=unused-argument, too-few-public-methods, missing-class-docstring, missing-function-docstring
# pylint: disable=no-self-use, superfluous-parens, no-else-return

"""Implementation of the recommender engine stub - for development purpose"""


import numpy as np


class RecommenderEngineStub:
    """stub class representing a recommender engine"""

    @classmethod
    def get_recommendations(cls, context):
        """This function allows to get recommended services for the recommendation
        sendpoint"""

        if context["panel_id"] == "version_A":
            return cls._get_services_ids(3)
        elif context["panel_id"] == "version_B":
            return cls._get_services_ids(2)
        else:
            raise InvalidRecommendationPanelIDError

    @classmethod
    def _get_services_ids(cls, services_number):
        return np.random.randint(
            low=0, high=10000, size=services_number, dtype=np.int32
        ).tolist()


class RecommendationEngineError(Exception):
    pass


class InvalidRecommendationPanelIDError(RecommendationEngineError):
    def message(self):
        return "Invalid recommendation panel id error"
