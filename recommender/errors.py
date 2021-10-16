# pylint: disable=missing-class-docstring, missing-function-docstring, no-self-use

"""This module contains error classes.
All custom exceptions should be placed in this file and this file should be
 placed directly in the recommender root directory to avoid cyclic import
 errors (notice that custom exceptions will be imported to the modules where
 they are raise but also to the modules where they will be caught).

"""


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


class DataValidationError(RecommendationEngineError):
    def message(self):
        return "Data provided in the pipeline are invalid"


class NCFDataValidationError(DataValidationError):
    def message(self):
        return "Data provided in the NCF pipeline are invalid"


class TooFewOrderedServicesError(NCFDataValidationError):
    def message(self):
        return "There is no user in the dataset that has at least 5 oredered services"


class NoUserInDatasetError(NCFDataValidationError):
    def message(self):
        return "There is no user in the dataset"


class ImbalancedDatasetError(NCFDataValidationError):
    def message(self):
        return "There is no user in the dataset"


class NoSavedMLComponentError(RecommendationEngineError):
    def message(self):
        return "No saved ML component"


class InferenceTooSlowError(RecommendationEngineError):
    def message(self):
        return "Inference time of the trained model is too long"


class PerformanceTooLowError(RecommendationEngineError):
    def message(self):
        return (
            "Inference performance of the trained model is too low"
            " regarding adopted metric"
        )
