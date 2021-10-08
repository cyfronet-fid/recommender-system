# pylint: disable=missing-class-docstring, missing-function-docstring, no-self-use

"""This module contains error classes"""


class RecommendationEngineError(Exception):
    pass


class InvalidRecommendationPanelIDError(RecommendationEngineError):
    def message(self):  # pragma: no cover
        return "Invalid recommendation panel id error."


class NoActorForK(RecommendationEngineError):
    def message(self):  # pragma: no cover
        return (
            "There is no Actor model for used K (number of services in"
            " recommendation)."
        )


class NoHistoryEmbedderForK(RecommendationEngineError):
    def message(self):  # pragma: no cover
        return (
            "There is no HistoryEmbedder for used K (number of services in"
            " recommendation)."
        )


class NoSearchPhraseEmbedderForK(RecommendationEngineError):
    def message(self):  # pragma: no cover
        return (
            "There is no SearchPhraseEmbedder for used K (number of services in"
            " recommendation)."
        )


class MissingComponentError(RecommendationEngineError):
    pass


class InsufficientRecommendationSpace(RecommendationEngineError):
    def message(self):  # pragma: no cover
        return (
            "The required number of services to recommend exceed the"
            " matching services space size."
        )


class MissingDependency(RecommendationEngineError):
    def message(self):  # pragma: no cover
        return "One or more of pipeline dependencies are missing."


class DifferentTypeObjectsInCollectionError(RecommendationEngineError):
    def message(self):
        return "All objects should be users or all objects should be services"


class MissingOneHotTensorError(RecommendationEngineError):
    def message(self):
        return "One or more objects don't have one hot tensor"


class MissingDenseTensorError(RecommendationEngineError):
    def message(self):
        return "One or more objects don't have dense tensor needed to"
