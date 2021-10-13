# pylint: disable=missing-module-docstring, missing-class-docstring, too-few-public-methods
# pylint: disable=unnecessary-pass, missing-function-docstring

"""Abstract pipeline steps"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple


class BaseStep(ABC):
    def __init__(self, config: Dict[str, object]):
        self.config = config

    @abstractmethod
    def __call__(self, data=None) -> Tuple[object, dict]:
        pass


class DataExtractionStep(BaseStep, ABC):
    def __init__(self, config):
        super().__init__(config)
        pass


class DataValidationStep(BaseStep, ABC):
    def __init__(self, config):
        super().__init__(config)
        pass


class DataPreparationStep(BaseStep, ABC):
    def __init__(self, config):
        super().__init__(config)
        pass


class ModelTrainingStep(BaseStep):
    def __init__(self, config):
        super().__init__(config)
        pass

    @abstractmethod
    def save(self):
        pass


class ModelEvaluationStep(BaseStep, ABC):
    def __init__(self, config):
        super().__init__(config)
        pass


class ModelValidationStep(BaseStep):
    def __init__(self, config):
        super().__init__(config)
        pass

    @abstractmethod
    def __call__(self, data=None) -> Tuple[object, object]:
        pass
