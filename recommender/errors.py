# pylint: disable=missing-class-docstring, missing-function-docstring, no-self-use

"""This module contains error classes"""


class RecommendationEngineError(Exception):
    pass


class InvalidRecommendationPanelIDError(RecommendationEngineError):
    def message(self):  # pragma: no cover
        return "Invalid recommendation panel id error"


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


class NoActionEmbedderForK(RecommendationEngineError):
    def message(self):  # pragma: no cover
        return (
            "There is no ActionEmbedder for used K (number of services in"
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
    def message(self):
        return (
            "The required number of services to recommend exceed the"
            " matching services space size"
        )
