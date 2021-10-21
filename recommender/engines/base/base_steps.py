# pylint: disable=missing-module-docstring, missing-class-docstring, too-few-public-methods
# pylint: disable=unnecessary-pass, missing-function-docstring

"""Abstract pipeline steps"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any


class BaseStep(ABC):
    def __init__(self, pipeline_config: Dict[str, Any]):
        self.pipeline_config = pipeline_config
        self.config = None

    @abstractmethod
    def __call__(self, data=None) -> Tuple[Any, Dict]:
        pass

    def resolve_constant(self, name, default=None):
        """step's config overrides pipeline config
        pipeline config overrides default"""

        return self.config.get(name, self.pipeline_config.get(name, default))


class DataExtractionStep(BaseStep, ABC):
    def __init__(self, pipeline_config):
        super().__init__(pipeline_config)
        self.config = self.pipeline_config[DataExtractionStep.__name__]


class DataValidationStep(BaseStep, ABC):
    def __init__(self, pipeline_config):
        super().__init__(pipeline_config)
        self.config = self.pipeline_config[DataValidationStep.__name__]


class DataPreparationStep(BaseStep, ABC):
    def __init__(self, pipeline_config):
        super().__init__(pipeline_config)
        self.config = self.pipeline_config[DataPreparationStep.__name__]


class ModelTrainingStep(BaseStep):
    def __init__(self, pipeline_config):
        super().__init__(pipeline_config)
        self.config = self.pipeline_config[ModelTrainingStep.__name__]

    @abstractmethod
    def save(self) -> None:
        """Save trained model"""


class ModelEvaluationStep(BaseStep, ABC):
    def __init__(self, pipeline_config):
        super().__init__(pipeline_config)
        self.config = self.pipeline_config[ModelEvaluationStep.__name__]


class ModelValidationStep(BaseStep, ABC):
    def __init__(self, pipeline_config):
        super().__init__(pipeline_config)
        self.config = self.pipeline_config[ModelValidationStep.__name__]
