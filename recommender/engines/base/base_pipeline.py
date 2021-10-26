# pylint: disable=too-few-public-methods

"""Abstract pipeline"""

from abc import ABC, abstractmethod
from typing import Dict
from datetime import datetime

from recommender.engines.base.base_steps import (
    DataExtractionStep,
    DataValidationStep,
    DataPreparationStep,
    ModelTrainingStep,
    ModelEvaluationStep,
    ModelValidationStep,
    BaseStep,
)
from recommender.engines.constants import VERBOSE
from recommender.models.pipeline_metadata import PipelineMetadata
from recommender.models.step_metadata import StepMetadata, Status


class BasePipeline(ABC):
    """Abstract Pipeline"""

    def __init__(self, pipeline_config: dict):
        self.pipeline_config = pipeline_config
        self.metadata = self._create_metadata()
        self.steps: Dict[str, BaseStep] = {
            DataExtractionStep.__name__: self._create_data_extraction_step(),
            DataValidationStep.__name__: self._create_data_validation_step(),
            DataPreparationStep.__name__: self._create_data_preparation_step(),
            ModelTrainingStep.__name__: self._create_model_training_step(),
            ModelEvaluationStep.__name__: self._create_model_evaluation_step(),
            ModelValidationStep.__name__: self._create_model_validation_step(),
        }
        self.verbose = self.pipeline_config.get(VERBOSE, True)

    def _create_metadata(self) -> PipelineMetadata:
        return PipelineMetadata(
            type=f"{self.__class__.__name__}",
        )

    @abstractmethod
    def _check_dependencies(self):
        """Raises MissingDependency if needed."""

    def __call__(self):
        print(f"Started {self.__class__.__name__}...")
        self._check_dependencies()

        self._update_metadata("start_time", datetime.utcnow())
        self._update_metadata(
            "steps",
            [
                StepMetadata(
                    type=f"{step.__class__.__name__}", status=Status.NOT_COMPLETED
                )
                for step in self.steps.values()
            ],
        )

        data = None
        for step_metadata, step in zip(self.metadata.steps, self.steps.values()):
            print(f"Started {step.__class__.__name__}...")
            step_metadata.start_time = datetime.utcnow()
            self._update_metadata("steps", self.metadata.steps)
            data, details = step(data)
            step_metadata.details = details
            step_metadata.end_time = datetime.utcnow()
            step_metadata.status = Status.COMPLETED

            if self.verbose:
                print(details)

            self._update_metadata("steps", self.metadata.steps)

            print(f"Finished {step.__class__.__name__}!")

        self.steps[ModelTrainingStep.__name__].save()
        self._update_metadata("end_time", datetime.utcnow())

        print(f"Finished {self.__class__.__name__}!")

    def _update_metadata(self, attr, value):
        setattr(self.metadata, attr, value)
        self.metadata.save()

    @abstractmethod
    def _create_data_extraction_step(self) -> DataExtractionStep:
        pass

    @abstractmethod
    def _create_data_validation_step(self) -> DataExtractionStep:
        pass

    @abstractmethod
    def _create_data_preparation_step(self) -> DataExtractionStep:
        pass

    @abstractmethod
    def _create_model_training_step(self) -> DataExtractionStep:
        pass

    @abstractmethod
    def _create_model_evaluation_step(self) -> DataExtractionStep:
        pass

    @abstractmethod
    def _create_model_validation_step(self) -> DataExtractionStep:
        pass
