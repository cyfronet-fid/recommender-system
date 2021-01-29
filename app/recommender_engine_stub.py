# pylint: disable=unused-argument, too-few-public-methods, missing-class-docstring, missing-function-docstring
# pylint: disable=no-self-use, superfluous-parens

"""Implementation of the recommender engine stub - for development purpose"""


import numpy as np


class RecommenderEngineStub:
    """stub class representing a recommender engine"""

    @classmethod
    def get_recommendations(cls, context, location, version):
        """This function allows to get recommended services for the recommendation \
        sendpoint"""
        if version not in ("v1", "v2"):
            raise InvalidRecommendationPanelVersionError

        if location not in ("services_list"):
            raise InvalidRecommendationPanelLocationError

        if version == "v1":
            action = cls._get_services_ids(3)
        elif version == "v2":
            action = cls._get_services_ids(2)

        return action

    @classmethod
    def _get_services_ids(cls, services_number):
        return np.random.randint(
            low=0, high=10000, size=services_number, dtype=np.int32
        ).tolist()


class RecommendationEngineError(Exception):
    pass


class InvalidRecommendationPanelVersionError(RecommendationEngineError):
    def message(self):
        return "Invalid recommendation panel version error"


class InvalidRecommendationPanelLocationError(RecommendationEngineError):
    def message(self):
        return "Invalid recommendation panel location error"
