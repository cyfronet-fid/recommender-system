# pylint: disable=too-few-public-methods

"""Abstract pipeline"""

from abc import ABC, abstractmethod
from typing import Dict
import time

from recommender.engines.base.base_steps import (
    DataExtractionStep,
    DataValidationStep,
    DataPreparationStep,
    ModelTrainingStep,
    ModelEvaluationStep,
    ModelValidationStep,
    BaseStep,
)
from recommender.models.pipeline_metadata import PipelineMetadata
from recommender.models.step_metadata import StepMetadata, Status


class BasePipeline(ABC):
    """Abstract Pipeline class"""

    def __init__(self, pipeline_config: dict):
        self.pipeline_config = pipeline_config
        self.metadata = self._create_metadata()
        self.steps: Dict[str, BaseStep] = {
            DataExtractionStep.__name__: self._create_data_extraction_step(
                pipeline_config[DataExtractionStep.__name__]
            ),
            DataValidationStep.__name__: self._create_data_validation_step(
                pipeline_config[DataValidationStep.__name__]
            ),
            DataPreparationStep.__name__: self._create_data_preparation_step(
                pipeline_config[DataPreparationStep.__name__]
            ),
            ModelTrainingStep.__name__: self._create_model_training_step(
                pipeline_config[ModelTrainingStep.__name__]
            ),
            ModelEvaluationStep.__name__: self._create_model_evaluation_step(
                pipeline_config[ModelEvaluationStep.__name__]
            ),
            ModelValidationStep.__name__: self._create_model_validation_step(
                pipeline_config[ModelValidationStep.__name__]
            ),
        }

    @abstractmethod
    def _create_metadata(self) -> PipelineMetadata:
        return PipelineMetadata(
            type=f"{self.__class__.__name__}",
        )

    @abstractmethod
    def _check_dependencies(self) -> bool:
        """Raises MissingDependency if needed."""

    def __call__(self):
        self._check_dependencies()

        self._update_metadata("start_time", time.time())
        self.metadata.steps = [
            StepMetadata(type=f"{step.__class__.__name__}", status=Status.NOT_COMPLETED)
            for step in self.steps.values()
        ]

        data = None
        for step_metadata, step in zip(self.metadata.steps, self.steps.values()):
            step_metadata.start_time = time.time()
            data, details = step(data)
            step_metadata.details = details
            step_metadata.end_time = time.time()
            step_metadata.status = Status.COMPLETED

            self._update_metadata("steps", self.metadata.steps)

        self.steps[ModelTrainingStep.__name__].save()
        self._update_metadata("end_time", time.time())

    def _update_metadata(self, attr, value):
        setattr(self.metadata, attr, value)
        self.metadata.save()

    @abstractmethod
    def _create_data_extraction_step(self, step_config) -> DataExtractionStep:
        pass

    @abstractmethod
    def _create_data_validation_step(self, step_config) -> DataExtractionStep:
        pass

    @abstractmethod
    def _create_data_preparation_step(self, step_config) -> DataExtractionStep:
        pass

    @abstractmethod
    def _create_model_training_step(self, step_config) -> DataExtractionStep:
        pass

    @abstractmethod
    def _create_model_evaluation_step(self, step_config) -> DataExtractionStep:
        pass

    @abstractmethod
    def _create_model_validation_step(self, step_config) -> DataExtractionStep:
        pass
