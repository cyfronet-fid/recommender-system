# pylint: disable=missing-class-docstring, missing-function-docstring, no-self-use

"""This module contains error classes"""

class RecommendationEngineError(Exception):
    pass


class InvalidRecommendationPanelIDError(RecommendationEngineError):
    def message(self):  # pragma: no cover
        return "Invalid recommendation panel id error"


class UntrainedPreAgentError(Exception):
    def message(self):  # pragma: no cover
        return (
            "Pre-Agent can't operate without trained Neural Collaborative"
            " Filtering model - train it before Pre-agent usage via "
            "'/training' endpoint"
        )


class TooSmallRecommendationSpace(RecommendationEngineError):
    def message(self):
        return "The required number of services to recommend exceed the" \
               " matching services space size"
